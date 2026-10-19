package main

import (
	"context"
	"errors"
	"testing"
)

func TestMemoryStore_NewIDsAreSequential(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx := context.Background()

	id1, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if id1 != "conv-001" {
		t.Fatalf("first id = %q, want conv-001", id1)
	}

	id2, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if id2 != "conv-002" {
		t.Fatalf("second id = %q, want conv-002", id2)
	}
}

func TestMemoryStore_NewConversationHasEmptyHistory(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx := context.Background()

	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	msgs, err := store.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if len(msgs) != 0 {
		t.Fatalf("Get() = %+v, want empty history", msgs)
	}
}

func TestMemoryStore_AppendGet(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx := context.Background()
	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}

	want := []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	if err := store.Append(ctx, id, want...); err != nil {
		t.Fatalf("Append() unexpected error: %v", err)
	}

	got, err := store.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if !messagesEqual(got, want) {
		t.Fatalf("Get() = %+v, want %+v", got, want)
	}

	got[0].Content = "mutated"
	again, err := store.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if again[0].Content != "hello" {
		t.Fatal("Get() returned a slice aliased to store memory")
	}
}

func TestMemoryStore_ConversationsAreIsolated(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx := context.Background()

	a, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	b, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}

	if err := store.Append(ctx, a, Message{Role: "user", Content: "my name is Carlos"}); err != nil {
		t.Fatalf("Append() unexpected error: %v", err)
	}

	msgsA, err := store.Get(ctx, a)
	if err != nil {
		t.Fatalf("Get(a) unexpected error: %v", err)
	}
	if len(msgsA) != 1 || msgsA[0].Content != "my name is Carlos" {
		t.Fatalf("Get(a) = %+v, want Carlos message", msgsA)
	}

	msgsB, err := store.Get(ctx, b)
	if err != nil {
		t.Fatalf("Get(b) unexpected error: %v", err)
	}
	if len(msgsB) != 0 {
		t.Fatalf("Get(b) = %+v, want empty (conversations must not share history)", msgsB)
	}
}

func TestMemoryStore_ListCreationOrder(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx := context.Background()

	ids, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List() unexpected error: %v", err)
	}
	if len(ids) != 0 {
		t.Fatalf("List() = %v, want empty", ids)
	}

	first, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	second, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}

	ids, err = store.List(ctx)
	if err != nil {
		t.Fatalf("List() unexpected error: %v", err)
	}
	want := []string{first, second}
	if len(ids) != 2 || ids[0] != want[0] || ids[1] != want[1] {
		t.Fatalf("List() = %v, want %v", ids, want)
	}
}

func TestMemoryStore_MissingConversation(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx := context.Background()

	_, err := store.Get(ctx, "conv-001")
	if !errors.Is(err, ErrConversationNotFound) {
		t.Fatalf("Get() error = %v, want ErrConversationNotFound", err)
	}

	err = store.Append(ctx, "conv-001", Message{Role: "user", Content: "hi"})
	if !errors.Is(err, ErrConversationNotFound) {
		t.Fatalf("Append() error = %v, want ErrConversationNotFound", err)
	}
}

func TestMemoryStore_CanceledContext(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx, cancel := context.WithCancel(context.Background())
	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	cancel()

	if _, err := store.New(ctx); err == nil {
		t.Fatal("New() error = nil, want canceled")
	}
	if _, err := store.Get(ctx, id); err == nil {
		t.Fatal("Get() error = nil, want canceled")
	}
	if _, err := store.List(ctx); err == nil {
		t.Fatal("List() error = nil, want canceled")
	}
	if err := store.Append(ctx, id, Message{Role: "user", Content: "hi"}); err == nil {
		t.Fatal("Append() error = nil, want canceled")
	}
}

func TestMemoryStore_AppendPreservesToolAssociation(t *testing.T) {
	t.Parallel()

	store := newMemoryStore()
	ctx := context.Background()
	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}

	history := []Message{
		{Role: "user", Content: "what is 2 plus 3?"},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{{
				Function: FunctionCall{Name: "add", Arguments: []byte(`{"a":2,"b":3}`)},
			}},
		},
		{Role: "tool", Content: "5"},
		{Role: "assistant", Content: "5"},
	}
	if err := store.Append(ctx, id, history...); err != nil {
		t.Fatalf("Append() unexpected error: %v", err)
	}

	got, err := store.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if !messagesEqual(got, history) {
		t.Fatalf("Get() = %+v, want tool calls associated in order", got)
	}
	if len(got[1].ToolCalls) != 1 || got[1].ToolCalls[0].Function.Name != "add" {
		t.Fatalf("assistant tool calls = %+v, want add", got[1].ToolCalls)
	}
	if got[2].Role != "tool" || got[2].Content != "5" {
		t.Fatalf("tool result = %+v, want content 5 after the matching call", got[2])
	}
}

func messagesEqual(a, b []Message) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Role != b[i].Role || a[i].Content != b[i].Content {
			return false
		}
		if len(a[i].ToolCalls) != len(b[i].ToolCalls) {
			return false
		}
		for j := range a[i].ToolCalls {
			if a[i].ToolCalls[j].Function.Name != b[i].ToolCalls[j].Function.Name {
				return false
			}
			if string(a[i].ToolCalls[j].Function.Arguments) != string(b[i].ToolCalls[j].Function.Arguments) {
				return false
			}
		}
	}
	return true
}
