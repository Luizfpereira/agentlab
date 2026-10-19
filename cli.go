package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
)

func runREPL(ctx context.Context, a *agent, stdin io.Reader, stdout io.Writer) error {
	fmt.Fprintln(stdout, "AgentLab")
	id, err := a.store.New(ctx)
	if err != nil {
		return err
	}
	printNewConversation(stdout, id)

	scanner := bufio.NewScanner(stdin)
	for {
		fmt.Fprint(stdout, "Você: ")
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return err
			}
			fmt.Fprintln(stdout)
			return nil
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case line == "/quit":
			return nil
		case line == "/new":
			id, err = a.store.New(ctx)
			if err != nil {
				return err
			}
			printNewConversation(stdout, id)
		case line == "/list":
			ids, err := a.store.List(ctx)
			if err != nil {
				return err
			}
			for _, listed := range ids {
				fmt.Fprintln(stdout, listed)
			}
		case strings.HasPrefix(line, "/open"):
			target, ok := parseOpenID(line)
			if !ok {
				fmt.Fprintln(stdout, "uso: /open ID")
				continue
			}
			if _, err := a.store.Get(ctx, target); err != nil {
				fmt.Fprintf(stdout, "Conversa %s não encontrada.\n", target)
				continue
			}
			id = target
			fmt.Fprintf(stdout, "Conversa %s restaurada.\n\n", id)
		case strings.HasPrefix(line, "/"):
			fmt.Fprintf(stdout, "comando desconhecido: %s\n", strings.Fields(line)[0])
		default:
			reply, err := a.run(ctx, id, line)
			if err != nil {
				fmt.Fprintf(stdout, "Erro: %v\n", err)
				continue
			}
			fmt.Fprintf(stdout, "Agente: %s\n\n", reply)
		}
	}
}

func printNewConversation(w io.Writer, id string) {
	fmt.Fprintf(w, "Nova conversa: %s\n\n", id)
}

func parseOpenID(line string) (string, bool) {
	fields := strings.Fields(line)
	if len(fields) != 2 || fields[0] != "/open" {
		return "", false
	}
	return fields[1], true
}
