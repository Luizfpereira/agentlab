package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

var ErrConversationNotFound = errors.New("conversation not found")

type ConversationStore interface {
	New(ctx context.Context) (id string, err error)
	Get(ctx context.Context, id string) ([]Message, error)
	List(ctx context.Context) ([]string, error)
	Append(ctx context.Context, id string, msgs ...Message) error
}

type memoryStore struct {
	mu      sync.Mutex
	nextID  int
	order   []string
	history map[string][]Message
}

func newMemoryStore() *memoryStore {
	return &memoryStore{
		nextID:  1,
		history: make(map[string][]Message),
	}
}

func (s *memoryStore) New(ctx context.Context) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	id := formatConversationID(s.nextID)
	s.nextID++
	s.order = append(s.order, id)
	s.history[id] = nil
	return id, nil
}

func (s *memoryStore) Get(ctx context.Context, id string) ([]Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	msgs, ok := s.history[id]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrConversationNotFound, id)
	}
	return cloneMessages(msgs), nil
}

func (s *memoryStore) List(ctx context.Context) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]string, len(s.order))
	copy(out, s.order)
	return out, nil
}

func (s *memoryStore) Append(ctx context.Context, id string, msgs ...Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.history[id]; !ok {
		return fmt.Errorf("%w: %s", ErrConversationNotFound, id)
	}
	s.history[id] = append(s.history[id], cloneMessages(msgs)...)
	return nil
}

func formatConversationID(n int) string {
	return fmt.Sprintf("conv-%03d", n)
}

func cloneMessages(msgs []Message) []Message {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]Message, len(msgs))
	copy(out, msgs)
	for i := range out {
		if len(out[i].ToolCalls) == 0 {
			continue
		}
		calls := make([]ToolCall, len(out[i].ToolCalls))
		copy(calls, out[i].ToolCalls)
		for j := range calls {
			if len(calls[j].Function.Arguments) > 0 {
				args := make([]byte, len(calls[j].Function.Arguments))
				copy(args, calls[j].Function.Arguments)
				calls[j].Function.Arguments = args
			}
		}
		out[i].ToolCalls = calls
	}
	return out
}
