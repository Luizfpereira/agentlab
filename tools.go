package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type Tool interface {
	Name() string
	Definition() ToolDefinition
	Execute(ctx context.Context, args json.RawMessage) (string, error)
}

type ToolDefinition struct {
	Type     string             `json:"type"`
	Function ToolFunctionSchema `json:"function"`
}

type ToolFunctionSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type Registry struct {
	tools map[string]Tool
}

func newRegistry(tools ...Tool) *Registry {
	reg := &Registry{tools: make(map[string]Tool, len(tools))}
	for _, tool := range tools {
		reg.tools[tool.Name()] = tool
	}
	return reg
}

func defaultRegistry() *Registry {
	return newRegistry(newAddTool())
}

func (r *Registry) Definitions() []ToolDefinition {
	defs := make([]ToolDefinition, 0, len(r.tools))
	for _, tool := range r.tools {
		defs = append(defs, tool.Definition())
	}
	return defs
}

func (r *Registry) Execute(ctx context.Context, name string, args json.RawMessage) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", fmt.Errorf("tool %q timed out: %w", name, err)
	}

	tool, ok := r.tools[name]
	if !ok {
		return "", fmt.Errorf("unknown tool %q", name)
	}

	raw, err := unwrapArgs(args)
	if err != nil {
		return "", err
	}
	if err := validateRequired(tool.Definition(), raw); err != nil {
		return "", err
	}

	result, err := tool.Execute(ctx, raw)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			return "", fmt.Errorf("tool %q timed out: %w", name, err)
		}
		return "", err
	}
	return result, nil
}

type paramSchema struct {
	Required []string `json:"required"`
}

func validateRequired(def ToolDefinition, args json.RawMessage) error {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(args, &obj); err != nil {
		return fmt.Errorf("invalid arguments: %w", err)
	}

	var schema paramSchema
	if len(def.Function.Parameters) > 0 {
		if err := json.Unmarshal(def.Function.Parameters, &schema); err != nil {
			return fmt.Errorf("invalid tool schema: %w", err)
		}
	}
	for _, field := range schema.Required {
		val, ok := obj[field]
		if !ok || string(bytes.TrimSpace(val)) == "null" {
			return fmt.Errorf("invalid arguments: %s is required", field)
		}
	}
	return nil
}

type addTool struct{}

func newAddTool() Tool {
	return addTool{}
}

func (addTool) Name() string { return "add" }

func (addTool) Definition() ToolDefinition {
	return ToolDefinition{
		Type: "function",
		Function: ToolFunctionSchema{
			Name:        "add",
			Description: `Add two numbers and return their sum.`,
			Parameters:  json.RawMessage(`{"type":"object","properties":{"a":{"type":"number","description":"First addend"},"b":{"type":"number","description":"Second addend"}},"required":["a","b"]}`),
		},
	}
}

func (addTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	var parsed struct {
		A *float64 `json:"a"`
		B *float64 `json:"b"`
	}
	if err := json.Unmarshal(args, &parsed); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if parsed.A == nil || parsed.B == nil {
		return "", fmt.Errorf("invalid arguments: a and b are required numbers")
	}

	return strconv.FormatFloat(*parsed.A+*parsed.B, 'f', -1, 64), nil
}

func unwrapArgs(args json.RawMessage) (json.RawMessage, error) {
	raw := bytes.TrimSpace(args)
	if len(raw) == 0 || string(raw) == "null" {
		return nil, fmt.Errorf("invalid arguments: empty")
	}
	if raw[0] == '"' {
		var inner string
		if err := json.Unmarshal(raw, &inner); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		raw = bytes.TrimSpace([]byte(inner))
		if len(raw) == 0 {
			return nil, fmt.Errorf("invalid arguments: empty")
		}
	}
	return raw, nil
}

func formatToolError(err error) string {
	return strings.TrimSpace(err.Error())
}
