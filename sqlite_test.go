package main

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"testing"
)

func TestSQLiteStore_NewGetList(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
	ctx := context.Background()

	ids, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List() unexpected error: %v", err)
	}
	if len(ids) != 0 {
		t.Fatalf("List() = %v, want empty", ids)
	}

	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if id != "conv-001" {
		t.Fatalf("New() = %q, want conv-001", id)
	}

	msgs, err := store.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if len(msgs) != 0 {
		t.Fatalf("Get() = %+v, want empty", msgs)
	}

	listed, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List() unexpected error: %v", err)
	}
	if len(listed) != 1 || listed[0] != id {
		t.Fatalf("List() = %v, want [%s]", listed, id)
	}
}

func TestSQLiteStore_ListCreationOrder(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
	ctx := context.Background()
	first, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	second, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	ids, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List() unexpected error: %v", err)
	}
	if len(ids) != 2 || ids[0] != first || ids[1] != second {
		t.Fatalf("List() = %v, want [%s %s] in creation order", ids, first, second)
	}
}

func TestSQLiteStore_ConversationsAreIsolated(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
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

	msgsB, err := store.Get(ctx, b)
	if err != nil {
		t.Fatalf("Get(b) unexpected error: %v", err)
	}
	if len(msgsB) != 0 {
		t.Fatalf("Get(b) = %+v, want empty", msgsB)
	}
}

func TestSQLiteStore_MissingConversation(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
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

func TestSQLiteStore_ReloadsToolHistory(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "agentlab.db")
	ctx := context.Background()
	history := []Message{
		{Role: "user", Content: "what is 2 plus 3?"},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{{
				Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":2,"b":3}`)},
			}},
		},
		{Role: "tool", Content: "5"},
		{Role: "assistant", Content: "5"},
	}

	store, err := openSQLite(path)
	if err != nil {
		t.Fatalf("openSQLite() unexpected error: %v", err)
	}
	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if err := store.Append(ctx, id, history...); err != nil {
		t.Fatalf("Append() unexpected error: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() unexpected error: %v", err)
	}

	reopened, err := openSQLite(path)
	if err != nil {
		t.Fatalf("reopen unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close() })

	got, err := reopened.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() after reopen unexpected error: %v", err)
	}
	if !messagesEqual(got, history) {
		t.Fatalf("Get() after reopen = %+v, want full tool history", got)
	}
	if len(got[1].ToolCalls) != 1 || got[1].ToolCalls[0].Function.Name != "add" {
		t.Fatalf("reloaded assistant tool calls = %+v", got[1].ToolCalls)
	}
	if got[2].Role != "tool" || got[2].Content != "5" {
		t.Fatalf("reloaded tool result = %+v, want content 5 associated with add", got[2])
	}

	ids, err := reopened.List(ctx)
	if err != nil {
		t.Fatalf("List() unexpected error: %v", err)
	}
	if len(ids) != 1 || ids[0] != id {
		t.Fatalf("List() after reopen = %v, want [%s]", ids, id)
	}
}

func TestSQLiteStore_PersistsPartialTurn(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "agentlab.db")
	ctx := context.Background()
	partial := []Message{
		{Role: "user", Content: "what is 10 plus 4?"},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{{
				Function: FunctionCall{Name: "add", Arguments: json.RawMessage(`{"a":10,"b":4}`)},
			}},
		},
		{Role: "tool", Content: "14"},
	}

	store, err := openSQLite(path)
	if err != nil {
		t.Fatalf("openSQLite() unexpected error: %v", err)
	}
	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if err := store.Append(ctx, id, partial...); err != nil {
		t.Fatalf("Append() unexpected error: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() unexpected error: %v", err)
	}

	reopened, err := openSQLite(path)
	if err != nil {
		t.Fatalf("reopen unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close() })

	got, err := reopened.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() unexpected error: %v", err)
	}
	if !messagesEqual(got, partial) {
		t.Fatalf("partial turn after reopen = %+v, want preserved prefix", got)
	}
}

func TestSQLiteStore_IDsContinueAfterReopen(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "agentlab.db")
	ctx := context.Background()

	store, err := openSQLite(path)
	if err != nil {
		t.Fatalf("openSQLite() unexpected error: %v", err)
	}
	if _, err := store.New(ctx); err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() unexpected error: %v", err)
	}

	reopened, err := openSQLite(path)
	if err != nil {
		t.Fatalf("reopen unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close() })

	id, err := reopened.New(ctx)
	if err != nil {
		t.Fatalf("New() after reopen unexpected error: %v", err)
	}
	if id != "conv-002" {
		t.Fatalf("New() after reopen = %q, want conv-002", id)
	}
}

func TestSQLiteStore_CanceledContext(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
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

func TestSQLiteStore_ClosedDB(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
	ctx := context.Background()
	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() unexpected error: %v", err)
	}

	if _, err := store.New(ctx); err == nil {
		t.Fatal("New() after Close() error = nil")
	}
	if _, err := store.Get(ctx, id); err == nil {
		t.Fatal("Get() after Close() error = nil")
	}
	if _, err := store.List(ctx); err == nil {
		t.Fatal("List() after Close() error = nil")
	}
	if err := store.Append(ctx, id, Message{Role: "user", Content: "hi"}); err == nil {
		t.Fatal("Append() after Close() error = nil")
	}
}

func TestSQLiteStore_CorruptHistory(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
	ctx := context.Background()
	id, err := store.New(ctx)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if _, err := store.db.ExecContext(ctx, `UPDATE conversations SET messages = 'not-json' WHERE id = ?`, id); err != nil {
		t.Fatalf("seed corrupt history: %v", err)
	}
	if _, err := store.Get(ctx, id); err == nil {
		t.Fatal("Get() error = nil, want decode error")
	}
	if err := store.Append(ctx, id, Message{Role: "user", Content: "hi"}); err == nil {
		t.Fatal("Append() error = nil, want decode error")
	}

	if _, err := store.db.ExecContext(ctx, `UPDATE conversations SET messages = '' WHERE id = ?`, id); err != nil {
		t.Fatalf("seed empty history: %v", err)
	}
	msgs, err := store.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get() empty JSON unexpected error: %v", err)
	}
	if len(msgs) != 0 {
		t.Fatalf("Get() = %+v, want empty", msgs)
	}
}

func TestSQLiteStore_NewSQLErrors(t *testing.T) {
	t.Parallel()

	store := openTempSQLite(t)
	ctx := context.Background()
	if _, err := store.New(ctx); err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	if _, err := store.db.ExecContext(ctx, `UPDATE meta SET value = '1' WHERE key = ?`, metaNextIDKey); err != nil {
		t.Fatalf("reset next_id: %v", err)
	}
	if _, err := store.New(ctx); err == nil {
		t.Fatal("New() duplicate id error = nil")
	}

	broken := openTempSQLite(t)
	if _, err := broken.db.ExecContext(ctx, `DROP TABLE meta`); err != nil {
		t.Fatalf("drop meta: %v", err)
	}
	if _, err := broken.New(ctx); err == nil {
		t.Fatal("New() missing meta error = nil")
	}
}

func TestSQLiteStore_OpenInvalidPath(t *testing.T) {
	t.Parallel()

	_, err := openSQLite(t.TempDir())
	if err == nil {
		t.Fatal("openSQLite(dir) error = nil, want failure")
	}
}

func openTempSQLite(t *testing.T) *sqliteStore {
	t.Helper()
	store, err := openSQLite(filepath.Join(t.TempDir(), "agentlab.db"))
	if err != nil {
		t.Fatalf("openSQLite() unexpected error: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}
