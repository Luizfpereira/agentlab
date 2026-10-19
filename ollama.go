package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	defaultOllamaURL = "http://localhost:11434"
	defaultModel     = "qwen3:8b"
	coderModel       = "qwen2:7b"
)

type OllamaRequest struct {
	Model    string           `json:"model"`
	Messages []Message        `json:"messages"`
	Tools    []ToolDefinition `json:"tools,omitempty"`
	Stream   bool             `json:"stream"`
}

type OllamaResponse struct {
	Message Message `json:"message"`
}

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func ask(ctx context.Context, client *http.Client, baseURL, instruction string) (string, error) {
	msg, err := chat(ctx, client, baseURL, []Message{{Role: "user", Content: instruction}}, nil)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(msg.Content), nil
}

func chat(ctx context.Context, client *http.Client, baseURL string, messages []Message, tools []ToolDefinition) (Message, error) {
	payload := OllamaRequest{
		Model:    coderModel,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return Message{}, fmt.Errorf("encode ollama request: %w", err)
	}

	url := strings.TrimRight(baseURL, "/") + "/api/chat"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Message{}, fmt.Errorf("create ollama request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("ollama is unreachable: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return Message{}, fmt.Errorf("read ollama response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Message{}, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, truncate(string(respBody), 256))
	}

	var decoded OllamaResponse
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		return Message{}, fmt.Errorf("decode ollama response: %w", err)
	}
	if len(decoded.Message.ToolCalls) == 0 && strings.TrimSpace(decoded.Message.Content) == "" {
		return Message{}, fmt.Errorf("decode ollama response: empty message content")
	}

	return decoded.Message, nil
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
