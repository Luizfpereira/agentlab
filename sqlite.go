package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "modernc.org/sqlite"
)

const (
	metaNextIDKey    = "next_id"
	emptyHistoryJSON = "[]"
)

var _ ConversationStore = (*sqliteStore)(nil)

type sqliteStore struct {
	db *sql.DB
}

func openSQLite(path string) (*sqliteStore, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	db.SetMaxOpenConns(1)

	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS meta (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL
		);
		CREATE TABLE IF NOT EXISTS conversations (
			id TEXT PRIMARY KEY,
			created_at TEXT NOT NULL,
			messages TEXT NOT NULL
		);
		INSERT INTO meta(key, value) VALUES (?, '1')
		ON CONFLICT(key) DO NOTHING;
	`, metaNextIDKey); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("init sqlite schema: %w", err)
	}

	return &sqliteStore{db: db}, nil
}

func (s *sqliteStore) Close() error {
	return s.db.Close()
}

func (s *sqliteStore) New(ctx context.Context) (string, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return "", err
	}
	defer func() { _ = tx.Rollback() }()

	var next int
	if err := tx.QueryRowContext(ctx, `SELECT value FROM meta WHERE key = ?`, metaNextIDKey).Scan(&next); err != nil {
		return "", fmt.Errorf("read next conversation id: %w", err)
	}

	id := formatConversationID(next)
	if _, err := tx.ExecContext(ctx,
		`INSERT INTO conversations(id, created_at, messages) VALUES (?, ?, ?)`,
		id, time.Now().UTC().Format(time.RFC3339Nano), emptyHistoryJSON,
	); err != nil {
		return "", fmt.Errorf("insert conversation: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `UPDATE meta SET value = ? WHERE key = ?`, next+1, metaNextIDKey); err != nil {
		return "", fmt.Errorf("update next conversation id: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	return id, nil
}

func (s *sqliteStore) Get(ctx context.Context, id string) ([]Message, error) {
	var raw string
	err := s.db.QueryRowContext(ctx, `SELECT messages FROM conversations WHERE id = ?`, id).Scan(&raw)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("%w: %s", ErrConversationNotFound, id)
	}
	if err != nil {
		return nil, err
	}
	return decodeHistory(raw)
}

func (s *sqliteStore) List(ctx context.Context) ([]string, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id FROM conversations ORDER BY rowid ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

func (s *sqliteStore) Append(ctx context.Context, id string, msgs ...Message) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	var raw string
	err = tx.QueryRowContext(ctx, `SELECT messages FROM conversations WHERE id = ?`, id).Scan(&raw)
	if err == sql.ErrNoRows {
		return fmt.Errorf("%w: %s", ErrConversationNotFound, id)
	}
	if err != nil {
		return err
	}

	history, err := decodeHistory(raw)
	if err != nil {
		return err
	}
	history = append(history, cloneMessages(msgs)...)
	encoded, err := json.Marshal(history)
	if err != nil {
		return fmt.Errorf("encode conversation history: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `UPDATE conversations SET messages = ? WHERE id = ?`, string(encoded), id); err != nil {
		return err
	}
	return tx.Commit()
}

func decodeHistory(raw string) ([]Message, error) {
	if raw == "" || raw == emptyHistoryJSON {
		return nil, nil
	}
	var msgs []Message
	if err := json.Unmarshal([]byte(raw), &msgs); err != nil {
		return nil, fmt.Errorf("decode conversation history: %w", err)
	}
	return msgs, nil
}
