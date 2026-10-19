package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestAsk_ValidQuestion(t *testing.T) {
	t.Parallel()

	const instruction = "what is 2+2?"
	const reply = "4"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/api/chat" {
			t.Errorf("path = %s, want /api/chat", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %s, want application/json", ct)
		}

		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.Model != coderModel {
			t.Errorf("model = %s, want %s", req.Model, coderModel)
		}
		if req.Stream {
			t.Error("stream = true, want false")
		}
		if len(req.Messages) != 1 || req.Messages[0].Role != "user" || req.Messages[0].Content != instruction {
			t.Errorf("messages = %+v, want user %q", req.Messages, instruction)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: reply},
		})
	}))
	defer srv.Close()

	got, err := ask(context.Background(), srv.Client(), srv.URL, instruction)
	if err != nil {
		t.Fatalf("ask() unexpected error: %v", err)
	}
	if got != reply {
		t.Fatalf("ask() = %q, want %q", got, reply)
	}
}

func TestAsk_OllamaDown(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()

	client := &http.Client{Timeout: 2 * time.Second}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := ask(ctx, client, url, "hello")
	if err == nil {
		t.Fatal("ask() error = nil, want unreachable error")
	}
	if !strings.Contains(err.Error(), "unreachable") {
		t.Fatalf("ask() error = %q, want it to mention unreachable", err)
	}
}

func TestAsk_HTTPError(t *testing.T) {
	t.Parallel()

	const fakeReply = "this must not be treated as a model reply"

	cases := []struct {
		name string
		body string
	}{
		{name: "short", body: fakeReply},
		{name: "long", body: strings.Repeat("x", 300) + fakeReply},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			got, err := ask(context.Background(), srv.Client(), srv.URL, "hello")
			if err == nil {
				t.Fatal("ask() error = nil, want HTTP status error")
			}
			if got == fakeReply {
				t.Fatal("ask() returned error-body content as a valid model reply")
			}
			if !strings.Contains(err.Error(), "500") {
				t.Fatalf("ask() error = %q, want status 500", err)
			}
			if len(tc.body) > 256 && strings.Contains(err.Error(), fakeReply) {
				t.Fatalf("ask() error dumped the full body: %q", err)
			}
		})
	}
}

func TestAsk_InvalidJSON(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{not-json`)
	}))
	defer srv.Close()

	_, err := ask(context.Background(), srv.Client(), srv.URL, "hello")
	if err == nil {
		t.Fatal("ask() error = nil, want decode error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "decode") {
		t.Fatalf("ask() error = %q, want decode error", err)
	}
}

func TestAsk_EmptyContent(t *testing.T) {
	t.Parallel()

	for _, content := range []string{"", "   "} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: content},
			})
		}))

		_, err := ask(context.Background(), srv.Client(), srv.URL, "hello")
		srv.Close()
		if err == nil {
			t.Fatalf("ask() error = nil for content %q, want empty content error", content)
		}
		if !strings.Contains(strings.ToLower(err.Error()), "decode") {
			t.Fatalf("ask() error = %q, want decode error", err)
		}
	}
}

func TestAsk_InvalidURL(t *testing.T) {
	t.Parallel()

	_, err := ask(context.Background(), &http.Client{}, "http://[", "hello")
	if err == nil {
		t.Fatal("ask() error = nil, want request creation error")
	}
	if !strings.Contains(err.Error(), "create ollama request") {
		t.Fatalf("ask() error = %q, want create ollama request", err)
	}
}
