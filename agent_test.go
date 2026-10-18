package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func agentWithLog(client *http.Client, url string, reg *Registry) (*agent, *bytes.Buffer) {
	var buf bytes.Buffer
	a := newAgent(client, url, reg)
	a.logger = slog.New(slog.NewJSONHandler(&buf, nil))
	return a, &buf
}

func TestAgent_NoToolCall(t *testing.T) {
	t.Parallel()

	const reply = "hello from the model"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if !hasAddTool(req.Tools) {
			t.Error("request missing add tool definition")
		}
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: reply},
		})
	}))
	defer srv.Close()

	a, logs := agentWithLog(srv.Client(), srv.URL, newRegistry(newAddTool()))
	got, err := a.run(context.Background(), "say hello")
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	if got != reply {
		t.Fatalf("run() = %q, want %q", got, reply)
	}
	if !strings.Contains(logs.String(), `"stop_reason":"final_answer"`) {
		t.Fatalf("log = %s, want stop_reason=final_answer", logs.String())
	}
	if !strings.Contains(logs.String(), `"model_calls":1`) {
		t.Fatalf("log = %s, want model_calls=1", logs.String())
	}
}

func TestAgent_ToolCallReturnedToModel(t *testing.T) {
	t.Parallel()

	const finalReply = "2 + 3 = 5"
	var calls int

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		calls++
		switch calls {
		case 1:
			if !hasAddTool(req.Tools) {
				t.Error("first request missing add tool definition")
			}
			if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
				t.Errorf("first request messages = %+v, want a single user message", req.Messages)
			}
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{{
						Function: FunctionCall{
							Name:      "add",
							Arguments: json.RawMessage(`{"a":2,"b":3}`),
						},
					}},
				},
			})
		case 2:
			if !hasAssistantToolCall(req.Messages, "add") {
				t.Errorf("second request missing assistant tool call, messages=%+v", req.Messages)
			}
			if !hasToolResult(req.Messages, "5") {
				t.Errorf("second request missing tool result 5, messages=%+v", req.Messages)
			}
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: finalReply},
			})
		default:
			t.Errorf("unexpected extra request #%d", calls)
			http.Error(w, "too many requests", http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	got, err := a.run(context.Background(), "what is 2 plus 3?")
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	if got != finalReply {
		t.Fatalf("run() = %q, want %q", got, finalReply)
	}
	if calls != 2 {
		t.Fatalf("ollama calls = %d, want 2", calls)
	}
}

func TestAgent_MultipleToolCalls(t *testing.T) {
	t.Parallel()

	var second OllamaRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if len(req.Messages) == 1 {
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{
						{Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":2,"b":3}`)}},
						{Function: FunctionCall{Name: "shell", Arguments: json.RawMessage(`{"cmd":"rm"}`)}},
						{Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":10,"b":1}`)}},
					},
				},
			})
			return
		}
		second = req
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "done"},
		})
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	got, err := a.run(context.Background(), "add some numbers")
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	if got != "done" {
		t.Fatalf("run() = %q, want done", got)
	}

	var toolMsgs []Message
	for _, msg := range second.Messages {
		if msg.Role == "tool" {
			toolMsgs = append(toolMsgs, msg)
		}
	}
	if len(toolMsgs) != 3 {
		t.Fatalf("tool messages = %d, want 3: %+v", len(toolMsgs), second.Messages)
	}
	if toolMsgs[0].Content != "5" {
		t.Fatalf("first tool result = %q, want 5", toolMsgs[0].Content)
	}
	if !strings.Contains(strings.ToLower(toolMsgs[1].Content), "unknown tool") {
		t.Fatalf("second tool result = %q, want unknown tool error", toolMsgs[1].Content)
	}
	if toolMsgs[2].Content != "11" {
		t.Fatalf("third tool result = %q, want 11", toolMsgs[2].Content)
	}
}

func TestAgent_RespectsIterationLimit(t *testing.T) {
	t.Parallel()

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					Function: FunctionCall{
						Name:      "add",
						Arguments: json.RawMessage(`{"a":1,"b":1}`),
					},
				}},
			},
		})
	}))
	defer srv.Close()

	a, logs := agentWithLog(srv.Client(), srv.URL, newRegistry(newAddTool()))
	a.maxIters = 2

	_, err := a.run(context.Background(), "keep adding")
	if err == nil {
		t.Fatal("run() error = nil, want iteration limit error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "iteration") {
		t.Fatalf("run() error = %q, want iteration limit", err)
	}
	if calls != 2 {
		t.Fatalf("ollama calls = %d, want 2", calls)
	}
	if !strings.Contains(logs.String(), `"stop_reason":"iteration_limit"`) {
		t.Fatalf("log = %s, want stop_reason=iteration_limit", logs.String())
	}
}

func TestAgent_CanceledContext(t *testing.T) {
	t.Parallel()

	a, logs := agentWithLog(&http.Client{}, "http://example.invalid", newRegistry(newAddTool()))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := a.run(ctx, "hello")
	if err == nil {
		t.Fatal("run() error = nil, want canceled context")
	}
	if !strings.Contains(logs.String(), `"stop_reason":"canceled"`) {
		t.Fatalf("log = %s, want stop_reason=canceled", logs.String())
	}
}

func TestAgent_Timeout(t *testing.T) {
	t.Parallel()

	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		<-req.Context().Done()
		return nil, req.Context().Err()
	})}
	a, logs := agentWithLog(client, "http://example.invalid", newRegistry(newAddTool()))
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	_, err := a.run(ctx, "hello")
	if err == nil {
		t.Fatal("run() error = nil, want timeout")
	}
	if !strings.Contains(logs.String(), `"stop_reason":"timeout"`) {
		t.Fatalf("log = %s, want stop_reason=timeout", logs.String())
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestAgent_ToolTimeoutIsReturnedToModel(t *testing.T) {
	t.Parallel()

	var second OllamaRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if len(req.Messages) == 1 {
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{{
						Function: FunctionCall{Name: "hang", Arguments: json.RawMessage(`{}`)},
					}},
				},
			})
			return
		}
		second = req
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "the tool timed out"},
		})
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(hangTool{delay: 200 * time.Millisecond}))
	a.toolTimeout = 20 * time.Millisecond

	got, err := a.run(context.Background(), "please hang")
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	if got != "the tool timed out" {
		t.Fatalf("run() = %q, want follow-up after tool timeout", got)
	}
	if !hasToolResultContaining(second.Messages, "timed out") {
		t.Fatalf("tool timeout was not sent back to the model, messages=%+v", second.Messages)
	}
}

func TestAgent_UnknownToolSendsErrorWithoutExecutingOthers(t *testing.T) {
	t.Parallel()

	spy := &spyTool{}
	var secondReq OllamaRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if len(req.Messages) == 1 {
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{{
						Function: FunctionCall{Name: "shell", Arguments: json.RawMessage(`{"cmd":"rm"}`)},
					}},
				},
			})
			return
		}
		secondReq = req
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "I cannot run that tool"},
		})
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(spy))
	got, err := a.run(context.Background(), "delete everything")
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	if spy.called {
		t.Fatal("unknown tool caused a registered tool to run")
	}
	if !hasToolResultContaining(secondReq.Messages, "unknown tool") {
		t.Fatalf("tool error was not sent back to the model, messages=%+v", secondReq.Messages)
	}
	if got != "I cannot run that tool" {
		t.Fatalf("run() = %q, want model follow-up", got)
	}
}

func hasAddTool(tools []ToolDefinition) bool {
	for _, tool := range tools {
		if tool.Type == "function" && tool.Function.Name == "add" {
			return true
		}
	}
	return false
}

func hasAssistantToolCall(messages []Message, name string) bool {
	for _, msg := range messages {
		if msg.Role != "assistant" {
			continue
		}
		for _, call := range msg.ToolCalls {
			if call.Function.Name == name {
				return true
			}
		}
	}
	return false
}

func hasToolResult(messages []Message, want string) bool {
	for _, msg := range messages {
		if msg.Role == "tool" && msg.Content == want {
			return true
		}
	}
	return false
}

func hasToolResultContaining(messages []Message, substr string) bool {
	for _, msg := range messages {
		if msg.Role == "tool" && strings.Contains(strings.ToLower(msg.Content), substr) {
			return true
		}
	}
	return false
}
