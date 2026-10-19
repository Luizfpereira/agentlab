package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

const (
	maxAgentIterations  = 5
	defaultAgentTimeout = 5 * time.Minute
	defaultChatTimeout  = 2 * time.Minute
	defaultToolTimeout  = 5 * time.Second
)

const SYSTEM_PROMPT = `You are a concise assistant.
If a provided function clearly matches the user's request, call it.
Otherwise answer yourself.
The user-facing reply must contain only the answer (for math: only the number).
No explanation of how you got it, no implementation details.`

var errIterationLimit = errors.New("agent iteration limit reached")

type stopReason string

const (
	stopFinalAnswer    stopReason = "final_answer"
	stopIterationLimit stopReason = "iteration_limit"
	stopTimeout        stopReason = "timeout"
	stopCanceled       stopReason = "canceled"
	stopStoreError     stopReason = "store_error"
	stopOllamaError    stopReason = "ollama_error"
)

var errStoreOp = errors.New("conversation store")

type toolCallRecord struct {
	Name     string `json:"name"`
	OK       bool   `json:"ok"`
	Error    string `json:"error,omitempty"`
	Duration string `json:"duration"`
}

type runStats struct {
	modelCalls int
	toolCalls  []toolCallRecord
}

type agent struct {
	client      *http.Client
	baseURL     string
	registry    *Registry
	store       ConversationStore
	maxIters    int
	toolTimeout time.Duration
	logger      *slog.Logger
}

func newAgent(client *http.Client, baseURL string, registry *Registry) *agent {
	return &agent{
		client:      client,
		baseURL:     baseURL,
		registry:    registry,
		store:       newMemoryStore(),
		maxIters:    maxAgentIterations,
		toolTimeout: defaultToolTimeout,
		logger:      slog.New(slog.DiscardHandler),
	}
}

func (a *agent) run(ctx context.Context, id, question string) (reply string, err error) {
	start := time.Now()
	stats := &runStats{}
	ctx, cancel := withDeadline(ctx, defaultAgentTimeout)
	defer cancel()
	defer func() {
		a.logRun(stats, time.Since(start), classifyStop(err), err)
	}()

	history, err := a.store.Get(ctx, id)
	if err != nil {
		return "", wrapStore(err)
	}

	userMsg := Message{Role: "user", Content: question}
	if err := a.store.Append(ctx, id, userMsg); err != nil {
		return "", wrapStore(err)
	}
	history = append(history, userMsg)

	tools := a.registry.Definitions()

	for range a.maxIters {
		if err := ctx.Err(); err != nil {
			if errors.Is(err, context.Canceled) {
				return "", fmt.Errorf("agent canceled: %w", err)
			}
			return "", fmt.Errorf("agent timed out: %w", err)
		}

		outgoing := make([]Message, 0, 1+len(history))
		outgoing = append(outgoing, Message{Role: "system", Content: SYSTEM_PROMPT})
		outgoing = append(outgoing, history...)

		stats.modelCalls++
		msg, err := chat(ctx, a.client, a.baseURL, outgoing, tools)
		if err != nil {
			return "", err
		}
		if len(msg.ToolCalls) == 0 {
			reply := strings.TrimSpace(msg.Content)
			if err := a.store.Append(ctx, id, Message{Role: "assistant", Content: reply}); err != nil {
				return "", wrapStore(err)
			}
			return reply, nil
		}

		if err := a.store.Append(ctx, id, msg); err != nil {
			return "", wrapStore(err)
		}
		history = append(history, msg)
		for _, call := range msg.ToolCalls {
			toolMsg := Message{
				Role:    "tool",
				Content: a.executeTool(ctx, stats, call),
			}
			if err := a.store.Append(ctx, id, toolMsg); err != nil {
				return "", wrapStore(err)
			}
			history = append(history, toolMsg)
		}
	}

	return "", fmt.Errorf("%w (%d)", errIterationLimit, a.maxIters)
}

func (a *agent) executeTool(ctx context.Context, stats *runStats, call ToolCall) string {
	start := time.Now()
	toolCtx, cancel := context.WithTimeout(ctx, a.toolTimeout)
	defer cancel()

	result, err := a.registry.Execute(toolCtx, call.Function.Name, call.Function.Arguments)
	rec := toolCallRecord{
		Name:     call.Function.Name,
		OK:       err == nil,
		Duration: time.Since(start).String(),
	}
	if err != nil {
		rec.Error = formatToolError(err)
		result = rec.Error
	}
	stats.toolCalls = append(stats.toolCalls, rec)
	return result
}

func (a *agent) logRun(stats *runStats, d time.Duration, reason stopReason, err error) {
	attrs := []any{
		slog.Int("model_calls", stats.modelCalls),
		slog.Int("tool_call_count", len(stats.toolCalls)),
		slog.Any("tool_calls", stats.toolCalls),
		slog.Int64("duration_ms", d.Milliseconds()),
		slog.String("stop_reason", string(reason)),
	}
	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}
	a.logger.Info("agent run", attrs...)
}

func classifyStop(err error) stopReason {
	if err == nil {
		return stopFinalAnswer
	}
	if errors.Is(err, errIterationLimit) {
		return stopIterationLimit
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return stopTimeout
	}
	if errors.Is(err, context.Canceled) {
		return stopCanceled
	}
	if errors.Is(err, errStoreOp) {
		return stopStoreError
	}
	return stopOllamaError
}

func wrapStore(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w: %w", errStoreOp, err)
}

func withDeadline(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, d)
}
