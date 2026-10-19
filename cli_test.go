package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCLI_NewListOpen(t *testing.T) {
	t.Parallel()

	var followUp OllamaRequest
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
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: "You have not told me your name."},
			})
		case 3:
			followUp = req
			_ = json.NewEncoder(w).Encode(OllamaResponse{
				Message: Message{Role: "assistant", Content: "28"},
			})
		default:
			t.Errorf("unexpected extra request #%d", calls)
			http.Error(w, "too many requests", http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	stdin := strings.NewReader(strings.Join([]string{
		"My name is Carlos",
		"/new",
		"What is my name?",
		"/list",
		"/open conv-001",
		"What is that times 2?",
		"/quit",
	}, "\n") + "\n")
	var stdout, stderr bytes.Buffer
	err := run(context.Background(), nil, srv.Client(), srv.URL, stdin, &stdout, &stderr, newMemoryStore())
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}

	out := stdout.String()
	if !strings.Contains(out, "AgentLab") {
		t.Fatalf("stdout missing AgentLab banner: %q", out)
	}
	if !strings.Contains(out, "Nova conversa: conv-001") {
		t.Fatalf("stdout missing first conversation: %q", out)
	}
	if !strings.Contains(out, "Nova conversa: conv-002") {
		t.Fatalf("stdout missing /new conversation: %q", out)
	}
	if !strings.Contains(out, "conv-001") || !strings.Contains(out, "conv-002") {
		t.Fatalf("stdout missing /list ids: %q", out)
	}
	if !strings.Contains(out, "Conversa conv-001 restaurada.") {
		t.Fatalf("stdout missing /open restore: %q", out)
	}
	if !strings.Contains(out, "Agente: Nice to meet you, Carlos!") {
		t.Fatalf("stdout missing first reply: %q", out)
	}
	if !strings.Contains(out, "Agente: You have not told me your name.") {
		t.Fatalf("stdout missing isolated second-conversation reply: %q", out)
	}
	if !strings.Contains(out, "Agente: 28") {
		t.Fatalf("stdout missing restored-conversation reply: %q", out)
	}
	if !hasMessage(followUp.Messages, "user", "My name is Carlos") {
		t.Fatalf("/open did not restore conv-001 history, messages=%+v", followUp.Messages)
	}
	if hasMessage(followUp.Messages, "user", "What is my name?") {
		t.Fatalf("restored conv-001 included conv-002 history, messages=%+v", followUp.Messages)
	}
	if calls != 3 {
		t.Fatalf("ollama calls = %d, want 3", calls)
	}
}

func TestCLI_SQLiteReloadsConversation(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "agentlab.db")
	var followUp OllamaRequest
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
				Message: Message{Role: "assistant", Content: "Nice to meet you, Carlos!"},
			})
			return
		}
		followUp = req
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "Carlos"},
		})
	}))
	defer srv.Close()

	store, err := openSQLite(path)
	if err != nil {
		t.Fatalf("openSQLite() unexpected error: %v", err)
	}
	var stdout, stderr bytes.Buffer
	err = run(context.Background(), nil, srv.Client(), srv.URL, strings.NewReader("My name is Carlos\n/quit\n"), &stdout, &stderr, store)
	if err != nil {
		t.Fatalf("first run() unexpected error: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() unexpected error: %v", err)
	}

	reopened, err := openSQLite(path)
	if err != nil {
		t.Fatalf("reopen unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	stdout.Reset()
	err = run(context.Background(), nil, srv.Client(), srv.URL, strings.NewReader("/open conv-001\nWhat is my name?\n/quit\n"), &stdout, &stderr, reopened)
	if err != nil {
		t.Fatalf("second run() unexpected error: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "Nova conversa: conv-002") {
		t.Fatalf("restart missing new conversation: %q", out)
	}
	if !strings.Contains(out, "Conversa conv-001 restaurada.") {
		t.Fatalf("restart missing /open: %q", out)
	}
	if !hasMessage(followUp.Messages, "user", "My name is Carlos") {
		t.Fatalf("sqlite reload lost history, messages=%+v", followUp.Messages)
	}
}

func TestCLI_OpenMissingKeepsCurrent(t *testing.T) {
	t.Parallel()

	var lastReq OllamaRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&lastReq); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "ok"},
		})
	}))
	defer srv.Close()

	stdin := strings.NewReader("hello\n/open conv-999\nagain\n/quit\n")
	var stdout, stderr bytes.Buffer
	err := run(context.Background(), nil, srv.Client(), srv.URL, stdin, &stdout, &stderr, newMemoryStore())
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	if !strings.Contains(stdout.String(), "não encontrada") {
		t.Fatalf("stdout = %q, want missing-conversation error", stdout.String())
	}
	if !hasMessage(lastReq.Messages, "user", "hello") {
		t.Fatalf("current conversation was lost after failed /open, messages=%+v", lastReq.Messages)
	}
}

func TestCLI_CommandsAndAgentError(t *testing.T) {
	t.Parallel()

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: "ok"},
		})
	}))
	defer srv.Close()

	stdin := strings.NewReader(strings.Join([]string{
		"",
		"/open",
		"/foo",
		"hello",
		"again",
		"/quit",
	}, "\n") + "\n")
	var stdout, stderr bytes.Buffer
	err := run(context.Background(), nil, srv.Client(), srv.URL, stdin, &stdout, &stderr, newMemoryStore())
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "uso: /open ID") {
		t.Fatalf("stdout missing /open usage: %q", out)
	}
	if !strings.Contains(out, "comando desconhecido: /foo") {
		t.Fatalf("stdout missing unknown command: %q", out)
	}
	if !strings.Contains(out, "Erro:") {
		t.Fatalf("stdout missing agent error: %q", out)
	}
	if !strings.Contains(out, "Agente: ok") {
		t.Fatalf("stdout missing follow-up after error: %q", out)
	}
}

func TestCLI_StoreNewFails(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	err := run(context.Background(), nil, &http.Client{}, "http://example.invalid", bytes.NewReader(nil), &stdout, &stderr, failNewStore{err: errStoreUnavailable})
	if err == nil {
		t.Fatal("run() error = nil, want store New error")
	}
}

func TestCLI_NewAndListErrors(t *testing.T) {
	t.Parallel()

	store := &failAfterStore{inner: newMemoryStore(), failNewAfter: 1, failList: true, err: errStoreUnavailable}
	var stdout, stderr bytes.Buffer
	err := run(context.Background(), nil, &http.Client{}, "http://example.invalid", strings.NewReader("/new\n"), &stdout, &stderr, store)
	if err == nil {
		t.Fatal("run(/new) error = nil, want store New error")
	}

	store = &failAfterStore{inner: newMemoryStore(), failList: true, err: errStoreUnavailable}
	stdout.Reset()
	err = run(context.Background(), nil, &http.Client{}, "http://example.invalid", strings.NewReader("/list\n"), &stdout, &stderr, store)
	if err == nil {
		t.Fatal("run(/list) error = nil, want store List error")
	}
}

func TestCLI_ScannerError(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	err := run(context.Background(), nil, &http.Client{}, "http://example.invalid", errReader{}, &stdout, &stderr, newMemoryStore())
	if err == nil {
		t.Fatal("run() error = nil, want scanner error")
	}
}

func TestRun_StoreNewFailsOneShot(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	err := run(context.Background(), []string{"hello"}, &http.Client{}, "http://example.invalid", bytes.NewReader(nil), &stdout, &stderr, failNewStore{err: errStoreUnavailable})
	if err == nil {
		t.Fatal("run() error = nil, want store New error")
	}
}

func TestCLI_EOFStartsAndExits(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	err := run(context.Background(), nil, &http.Client{}, "http://example.invalid", bytes.NewReader(nil), &stdout, &stderr, newMemoryStore())
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "AgentLab") {
		t.Fatalf("stdout = %q, want AgentLab banner", out)
	}
	if !strings.Contains(out, "Nova conversa: conv-001") {
		t.Fatalf("stdout = %q, want new conversation", out)
	}
}

func TestRun_ValidQuestion(t *testing.T) {
	t.Parallel()

	const reply = "the sky is blue"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(OllamaResponse{
			Message: Message{Role: "assistant", Content: reply},
		})
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	err := run(context.Background(), []string{"why", "is", "the", "sky", "blue?"}, srv.Client(), srv.URL, bytes.NewReader(nil), &stdout, &stderr, newMemoryStore())
	if err != nil {
		t.Fatalf("run() unexpected error: %v", err)
	}
	if strings.TrimSpace(stdout.String()) != reply {
		t.Fatalf("stdout = %q, want %q", stdout.String(), reply)
	}
	if !strings.Contains(stderr.String(), `"stop_reason":"final_answer"`) {
		t.Fatalf("stderr = %q, want stop_reason final_answer", stderr.String())
	}
}

func TestRun_PropagatesAskError(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()

	var stdout, stderr bytes.Buffer
	err := run(context.Background(), []string{"hello"}, &http.Client{Timeout: 2 * time.Second}, url, bytes.NewReader(nil), &stdout, &stderr, newMemoryStore())
	if err == nil {
		t.Fatal("run() error = nil, want ask error")
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
	if !strings.Contains(stderr.String(), `"stop_reason":"ollama_error"`) {
		t.Fatalf("stderr = %q, want stop_reason ollama_error", stderr.String())
	}
}

type failNewStore struct {
	err error
}

func (s failNewStore) New(context.Context) (string, error) { return "", s.err }
func (failNewStore) Get(context.Context, string) ([]Message, error) {
	return nil, ErrConversationNotFound
}
func (failNewStore) List(context.Context) ([]string, error) { return nil, nil }
func (failNewStore) Append(context.Context, string, ...Message) error {
	return ErrConversationNotFound
}

type failAfterStore struct {
	inner        ConversationStore
	news         int
	failNewAfter int
	failList     bool
	err          error
}

func (s *failAfterStore) New(ctx context.Context) (string, error) {
	s.news++
	if s.failNewAfter > 0 && s.news > s.failNewAfter {
		return "", s.err
	}
	return s.inner.New(ctx)
}

func (s *failAfterStore) Get(ctx context.Context, id string) ([]Message, error) {
	return s.inner.Get(ctx, id)
}

func (s *failAfterStore) List(ctx context.Context) ([]string, error) {
	if s.failList {
		return nil, s.err
	}
	return s.inner.List(ctx)
}

func (s *failAfterStore) Append(ctx context.Context, id string, msgs ...Message) error {
	return s.inner.Append(ctx, id, msgs...)
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) {
	return 0, errStoreUnavailable
}
