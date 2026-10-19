package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

var errStoreUnavailable = errors.New("store unavailable")

func agentWithLog(client *http.Client, url string, reg *Registry) (*agent, *bytes.Buffer) {
	var buf bytes.Buffer
	a := newAgent(client, url, reg)
	a.logger = slog.New(slog.NewJSONHandler(&buf, nil))
	return a, &buf
}

func mustConv(t *testing.T, store ConversationStore) string {
	t.Helper()
	id, err := store.New(context.Background())
	if err != nil {
		t.Fatalf("store.New: %v", err)
	}
	return id
}

func isFirstModelCall(msgs []Message) bool {
	for _, msg := range msgs {
		if msg.Role == "assistant" || msg.Role == "tool" {
			return false
		}
	}
	return true
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
	got, err := a.run(context.Background(), mustConv(t, a.store), "say hello")
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
			if !isFirstModelCall(req.Messages) {
				t.Errorf("first request messages = %+v, want system+user only", req.Messages)
			}
			if !hasRole(req.Messages, "system") || !hasRole(req.Messages, "user") {
				t.Errorf("first request messages = %+v, want system and user", req.Messages)
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
	got, err := a.run(context.Background(), mustConv(t, a.store), "what is 2 plus 3?")
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
		if isFirstModelCall(req.Messages) {
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
	got, err := a.run(context.Background(), mustConv(t, a.store), "add some numbers")
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

	_, err := a.run(context.Background(), mustConv(t, a.store), "keep adding")
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

	id := mustConv(t, a.store)
	_, err := a.run(ctx, id, "hello")
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

	id := mustConv(t, a.store)
	_, err := a.run(ctx, id, "hello")
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

func TestAgent_ToolFailureReturnedToModel(t *testing.T) {
	t.Parallel()

	const finalReply = "I could not use the tool"
	const toolErr = "tool execution failed"
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
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{{
						Function: FunctionCall{
							Name:      "fail",
							Arguments: json.RawMessage(`{}`),
						},
					}},
				},
			})
		case 2:
			if !hasAssistantToolCall(req.Messages, "fail") {
				t.Errorf("second request missing assistant tool call, messages=%+v", req.Messages)
			}
			if !hasToolResultContaining(req.Messages, toolErr) {
				t.Errorf("second request missing tool error %q, messages=%+v", toolErr, req.Messages)
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

	a := newAgent(srv.Client(), srv.URL, newRegistry(failingTool{err: toolErr}))
	got, err := a.run(context.Background(), mustConv(t, a.store), "please fail")
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

func TestAgent_ToolTimeoutIsReturnedToModel(t *testing.T) {
	t.Parallel()

	var second OllamaRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if isFirstModelCall(req.Messages) {
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

	got, err := a.run(context.Background(), mustConv(t, a.store), "please hang")
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
		if isFirstModelCall(req.Messages) {
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
	got, err := a.run(context.Background(), mustConv(t, a.store), "delete everything")
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

func TestAgent_SameConversationKeepsContext(t *testing.T) {
	t.Parallel()

	var second OllamaRequest
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if bodyHasConversationID(req) {
			t.Error("conversation_id must not be sent to the model")
		}

		calls++
		switch calls {
		case 1:
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: "Nice to meet you, Carlos!"},
			})
		case 2:
			second = req
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: "Carlos"},
			})
		default:
			t.Errorf("unexpected extra request #%d", calls)
			http.Error(w, "too many requests", http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	id := mustConv(t, a.store)

	if _, err := a.run(context.Background(), id, "My name is Carlos"); err != nil {
		t.Fatalf("first run() unexpected error: %v", err)
	}
	got, err := a.run(context.Background(), id, "What is my name?")
	if err != nil {
		t.Fatalf("second run() unexpected error: %v", err)
	}
	if got != "Carlos" {
		t.Fatalf("run() = %q, want Carlos", got)
	}
	if !hasMessage(second.Messages, "user", "My name is Carlos") {
		t.Fatalf("second request missing first user turn, messages=%+v", second.Messages)
	}
	if !hasMessage(second.Messages, "assistant", "Nice to meet you, Carlos!") {
		t.Fatalf("second request missing first assistant turn, messages=%+v", second.Messages)
	}
	if !hasMessage(second.Messages, "user", "What is my name?") {
		t.Fatalf("second request missing current user turn, messages=%+v", second.Messages)
	}
}

func TestAgent_DifferentConversationsDoNotShareHistory(t *testing.T) {
	t.Parallel()

	var secondConvFirstReq OllamaRequest
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		calls++
		switch calls {
		case 1:
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: "Nice to meet you, Carlos!"},
			})
		case 2:
			secondConvFirstReq = req
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: "You have not told me your name."},
			})
		default:
			t.Errorf("unexpected extra request #%d", calls)
			http.Error(w, "too many requests", http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	idA := mustConv(t, a.store)
	idB := mustConv(t, a.store)

	if _, err := a.run(context.Background(), idA, "My name is Carlos"); err != nil {
		t.Fatalf("run(A) unexpected error: %v", err)
	}
	if _, err := a.run(context.Background(), idB, "What is my name?"); err != nil {
		t.Fatalf("run(B) unexpected error: %v", err)
	}
	if hasMessage(secondConvFirstReq.Messages, "user", "My name is Carlos") {
		t.Fatalf("conversation B received A's history: %+v", secondConvFirstReq.Messages)
	}
}

func TestAgent_ToolResultsStayAssociated(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if isFirstModelCall(req.Messages) {
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{
						{Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":2,"b":3}`)}},
						{Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":10,"b":1}`)}},
					},
				},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "done"},
		})
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	id := mustConv(t, a.store)
	if _, err := a.run(context.Background(), id, "add some numbers"); err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}

	got, err := a.store.Get(context.Background(), id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("history len = %d, want 5 (user, assistant tools, 2 tool results, final): %+v", len(got), got)
	}
	if got[0].Role != "user" {
		t.Fatalf("history[0] = %+v, want user", got[0])
	}
	if got[1].Role != "assistant" || len(got[1].ToolCalls) != 2 {
		t.Fatalf("history[1] = %+v, want assistant with 2 tool calls", got[1])
	}
	if got[2].Role != "tool" || got[2].Content != "5" {
		t.Fatalf("history[2] = %+v, want tool result 5 for first call", got[2])
	}
	if got[3].Role != "tool" || got[3].Content != "11" {
		t.Fatalf("history[3] = %+v, want tool result 11 for second call", got[3])
	}
	if got[4].Role != "assistant" || got[4].Content != "done" {
		t.Fatalf("history[4] = %+v, want final assistant", got[4])
	}
	for _, msg := range got {
		if msg.Role == "system" {
			t.Fatal("system prompt must not be stored in conversation history")
		}
	}
}

func TestAgent_PreservesHistoryOnRunFailure(t *testing.T) {
	t.Parallel()

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		calls++
		if calls == 1 {
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{{
						Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":10,"b":4}`)},
					}},
				},
			})
			return
		}
		http.Error(w, "ollama down", http.StatusInternalServerError)
	}))
	defer srv.Close()

	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	id := mustConv(t, a.store)
	_, err := a.run(context.Background(), id, "what is 10 plus 4?")
	if err == nil {
		t.Fatal("run() error = nil, want ollama error")
	}

	got, err := a.store.Get(context.Background(), id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("history len = %d, want 3 (user, assistant tool call, tool result): %+v", len(got), got)
	}
	if got[0].Role != "user" || got[0].Content != "what is 10 plus 4?" {
		t.Fatalf("history[0] = %+v, want user question", got[0])
	}
	if got[1].Role != "assistant" || !hasAssistantToolCall(got, "add") {
		t.Fatalf("history[1] = %+v, want assistant add call", got[1])
	}
	if got[2].Role != "tool" || got[2].Content != "14" {
		t.Fatalf("history[2] = %+v, want tool result 14 associated with the call", got[2])
	}
}

func TestAgent_AppendUserFailure(t *testing.T) {
	t.Parallel()

	inner := newMemoryStore()
	id := mustConv(t, inner)
	a, logs := agentWithLog(&http.Client{}, "http://example.invalid", newRegistry(newAddTool()))
	a.store = &hookStore{inner: inner, failAppend: 1, err: errStoreUnavailable}

	_, err := a.run(context.Background(), id, "hello")
	if err == nil {
		t.Fatal("run() error = nil, want append error")
	}
	if !strings.Contains(logs.String(), `"stop_reason":"store_error"`) {
		t.Fatalf("log = %s, want stop_reason=store_error", logs.String())
	}
}

func TestAgent_AppendFinalAnswerFailure(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "hi"},
		})
	}))
	defer srv.Close()

	inner := newMemoryStore()
	id := mustConv(t, inner)
	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	a.store = &hookStore{inner: inner, failAppend: 2, err: errStoreUnavailable}

	_, err := a.run(context.Background(), id, "hello")
	if err == nil {
		t.Fatal("run() error = nil, want append error")
	}
}

func TestAgent_AppendToolCallFailure(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":1,"b":1}`)},
				}},
			},
		})
	}))
	defer srv.Close()

	inner := newMemoryStore()
	id := mustConv(t, inner)
	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	a.store = &hookStore{inner: inner, failAppend: 2, err: errStoreUnavailable}

	_, err := a.run(context.Background(), id, "add")
	if err == nil {
		t.Fatal("run() error = nil, want append error")
	}
}

func TestAgent_AppendToolResultFailure(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":1,"b":1}`)},
				}},
			},
		})
	}))
	defer srv.Close()

	inner := newMemoryStore()
	id := mustConv(t, inner)
	a := newAgent(srv.Client(), srv.URL, newRegistry(newAddTool()))
	a.store = &hookStore{inner: inner, failAppend: 3, err: errStoreUnavailable}

	_, err := a.run(context.Background(), id, "add")
	if err == nil {
		t.Fatal("run() error = nil, want append error")
	}
}

func TestAgent_CanceledAfterToolCall(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cancel()
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":1,"b":1}`)},
				}},
			},
		})
	}))
	defer srv.Close()

	a, logs := agentWithLog(srv.Client(), srv.URL, newRegistry(newAddTool()))
	inner := a.store
	id := mustConv(t, inner)
	a.store = &hookStore{inner: inner}
	_, err := a.run(ctx, id, "add")
	if err == nil {
		t.Fatal("run() error = nil, want canceled")
	}
	if !strings.Contains(logs.String(), `"stop_reason":"canceled"`) {
		t.Fatalf("log = %s, want stop_reason=canceled", logs.String())
	}
}

func TestAgent_TimeoutAfterToolCall(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					Function: FunctionCall{Name: "hang", Arguments: json.RawMessage(`{}`)},
				}},
			},
		})
	}))
	defer srv.Close()

	a, logs := agentWithLog(srv.Client(), srv.URL, newRegistry(hangTool{delay: 200 * time.Millisecond}))
	a.toolTimeout = 200 * time.Millisecond
	inner := a.store
	id := mustConv(t, inner)
	a.store = &hookStore{inner: inner}
	_, err := a.run(ctx, id, "hang")
	if err == nil {
		t.Fatal("run() error = nil, want timeout")
	}
	if !strings.Contains(logs.String(), `"stop_reason":"timeout"`) {
		t.Fatalf("log = %s, want stop_reason=timeout", logs.String())
	}
}

func TestAgent_UnknownConversation(t *testing.T) {
	t.Parallel()

	a, logs := agentWithLog(&http.Client{}, "http://example.invalid", newRegistry(newAddTool()))
	_, err := a.run(context.Background(), "conv-999", "hello")
	if err == nil {
		t.Fatal("run() error = nil, want missing conversation")
	}
	if !strings.Contains(logs.String(), `"stop_reason":"store_error"`) {
		t.Fatalf("log = %s, want stop_reason=store_error", logs.String())
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

func hasRole(messages []Message, role string) bool {
	for _, msg := range messages {
		if msg.Role == role {
			return true
		}
	}
	return false
}

func hasMessage(messages []Message, role, content string) bool {
	for _, msg := range messages {
		if msg.Role == role && msg.Content == content {
			return true
		}
	}
	return false
}

func bodyHasConversationID(req OllamaRequest) bool {
	raw, err := json.Marshal(req)
	if err != nil {
		return false
	}
	return strings.Contains(string(raw), "conversation_id")
}

type hookStore struct {
	inner      ConversationStore
	failAppend int
	appendN    int
	err        error
}

func (s *hookStore) New(context.Context) (string, error) {
	return s.inner.New(context.Background())
}

func (s *hookStore) Get(_ context.Context, id string) ([]Message, error) {
	return s.inner.Get(context.Background(), id)
}

func (s *hookStore) List(context.Context) ([]string, error) {
	return s.inner.List(context.Background())
}

func (s *hookStore) Append(_ context.Context, id string, msgs ...Message) error {
	s.appendN++
	if s.failAppend > 0 && s.appendN == s.failAppend {
		return s.err
	}
	return s.inner.Append(context.Background(), id, msgs...)
}
