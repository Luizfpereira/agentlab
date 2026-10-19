package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestAdd_Valid(t *testing.T) {
	t.Parallel()

	got, err := newAddTool().Execute(context.Background(), json.RawMessage(`{"a":2,"b":3}`))
	if err != nil {
		t.Fatalf("Execute() unexpected error: %v", err)
	}
	if got != "5" {
		t.Fatalf("Execute() = %q, want 5", got)
	}
}

func TestAdd_CanceledContext(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := newAddTool().Execute(ctx, json.RawMessage(`{"a":2,"b":3}`))
	if err == nil {
		t.Fatal("Execute() error = nil, want canceled context")
	}
}

func TestAdd_InvalidJSON(t *testing.T) {
	t.Parallel()

	_, err := newRegistry(newAddTool()).Execute(context.Background(), "add", json.RawMessage(`{not-json`))
	if err == nil {
		t.Fatal("Execute() error = nil, want invalid arguments error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "invalid arguments") {
		t.Fatalf("Execute() error = %q, want invalid arguments", err)
	}
}

func TestAdd_StringArguments(t *testing.T) {
	t.Parallel()

	got, err := newRegistry(newAddTool()).Execute(context.Background(), "add", json.RawMessage(`"{\"a\":2,\"b\":3}"`))
	if err != nil {
		t.Fatalf("Execute() unexpected error: %v", err)
	}
	if got != "5" {
		t.Fatalf("Execute() = %q, want 5", got)
	}
}

func TestAdd_Floats(t *testing.T) {
	t.Parallel()

	got, err := newAddTool().Execute(context.Background(), json.RawMessage(`{"a":1.5,"b":2.25}`))
	if err != nil {
		t.Fatalf("Execute() unexpected error: %v", err)
	}
	if got != "3.75" {
		t.Fatalf("Execute() = %q, want 3.75", got)
	}
}

func TestAdd_EmptyArgs(t *testing.T) {
	t.Parallel()

	reg := newRegistry(newAddTool())
	for _, args := range []json.RawMessage{nil, json.RawMessage(""), json.RawMessage("null"), json.RawMessage(`""`), json.RawMessage(`"`)} {
		_, err := reg.Execute(context.Background(), "add", args)
		if err == nil {
			t.Fatalf("Execute(%s) error = nil, want invalid arguments", args)
		}
	}
}

func TestRegistry_MissingRequiredDoesNotExecute(t *testing.T) {
	t.Parallel()

	rec := &recordingAdd{}
	_, err := newRegistry(rec).Execute(context.Background(), "add", json.RawMessage(`{"a":2}`))
	if err == nil {
		t.Fatal("Execute() error = nil, want missing field error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "invalid arguments") {
		t.Fatalf("Execute() error = %q, want invalid arguments", err)
	}
	if rec.called {
		t.Fatal("tool ran before argument validation")
	}
}

func TestRegistry_UnknownTool(t *testing.T) {
	t.Parallel()

	spy := &spyTool{}
	_, err := newRegistry(spy).Execute(context.Background(), "rm", json.RawMessage(`{}`))
	if err == nil {
		t.Fatal("Execute() error = nil, want unknown tool error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "unknown tool") {
		t.Fatalf("Execute() error = %q, want unknown tool", err)
	}
	if spy.called {
		t.Fatal("unknown tool name executed a registered function")
	}
}

func TestRegistry_ToolTimeout(t *testing.T) {
	t.Parallel()

	reg := newRegistry(hangTool{delay: 200 * time.Millisecond})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err := reg.Execute(ctx, "hang", json.RawMessage(`{}`))
	if err == nil {
		t.Fatal("Execute() error = nil, want timeout")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "timed out") {
		t.Fatalf("Execute() error = %q, want timed out", err)
	}
}

type spyTool struct {
	called bool
}

func (s *spyTool) Name() string { return "spy" }

func (s *spyTool) Definition() ToolDefinition {
	return ToolDefinition{Type: "function", Function: ToolFunctionSchema{Name: "spy"}}
}

func (s *spyTool) Execute(context.Context, json.RawMessage) (string, error) {
	s.called = true
	return "executed", nil
}

type recordingAdd struct {
	addTool
	called bool
}

func (r *recordingAdd) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	r.called = true
	return r.addTool.Execute(ctx, args)
}

type hangTool struct {
	delay time.Duration
}

func (hangTool) Name() string { return "hang" }

func (hangTool) Definition() ToolDefinition {
	return ToolDefinition{
		Type:     "function",
		Function: ToolFunctionSchema{Name: "hang", Parameters: json.RawMessage(`{"type":"object"}`)},
	}
}

func (h hangTool) Execute(ctx context.Context, _ json.RawMessage) (string, error) {
	timer := time.NewTimer(h.delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-timer.C:
		return "hung", nil
	}
}

type failingTool struct {
	err string
}

func (failingTool) Name() string { return "fail" }

func (failingTool) Definition() ToolDefinition {
	return ToolDefinition{
		Type:     "function",
		Function: ToolFunctionSchema{Name: "fail", Parameters: json.RawMessage(`{"type":"object"}`)},
	}
}

func (f failingTool) Execute(context.Context, json.RawMessage) (string, error) {
	return "", errors.New(f.err)
}
