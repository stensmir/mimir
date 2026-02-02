# Claude Code Tasks: Руководство для разработчиков

## Что это

Система задач (Tasks) в Claude Code — встроенный трекер для управления сложными задачами. Агент сам создаёт, отслеживает и завершает задачи, понимая зависимости между ними.

## Быстрый старт

### Запуск с персистентными задачами

```bash
# Задачи сохраняются между сессиями
CLAUDE_CODE_TASK_LIST_ID=my-project claude
```

Без этой переменной задачи живут только в текущей сессии.

### Общий список для нескольких агентов

```bash
# Терминал 1
CLAUDE_CODE_TASK_LIST_ID=feature-auth claude

# Терминал 2 (видит те же задачи)
CLAUDE_CODE_TASK_LIST_ID=feature-auth claude
```

До 5 воркеров могут работать с одним списком одновременно.

## Инструменты

| Инструмент | Назначение |
|------------|------------|
| `TaskCreate` | Создать задачу с описанием |
| `TaskList` | Показать все задачи |
| `TaskGet` | Получить детали задачи по ID |
| `TaskUpdate` | Изменить статус, добавить зависимости |

## Статусы и зависимости

### Статусы

- `pending` — ожидает выполнения
- `in_progress` — в работе
- `completed` — завершена

### Зависимости

```
1. [completed] Настроить структуру проекта
2. [in_progress] Создать схему БД
3. [pending] Реализовать API (blocked by #2)
```

Поля `blockedBy` и `blocks` определяют порядок выполнения. Агент не возьмётся за заблокированную задачу.

## Несколько списков задач

Каждый `TASK_LIST_ID` создаёт отдельный независимый список. Можно работать над несколькими фичами параллельно:

```bash
# Команда 1: авторизация (2 разработчика)
CLAUDE_CODE_TASK_LIST_ID=feature-auth claude      # Терминал 1
CLAUDE_CODE_TASK_LIST_ID=feature-auth claude      # Терминал 2

# Команда 2: платежи (2 разработчика)
CLAUDE_CODE_TASK_LIST_ID=feature-payments claude  # Терминал 3
CLAUDE_CODE_TASK_LIST_ID=feature-payments claude  # Терминал 4
```

**Результат:**
- Терминалы 1-2 видят и редактируют список `feature-auth`
- Терминалы 3-4 видят и редактируют список `feature-payments`
- Списки полностью изолированы друг от друга

**Хранение:**

```
~/.claude/
├── tasks-feature-auth.json
├── tasks-feature-payments.json
└── tasks-refactor-api.json
```

## Конфигурация

### Способ 1: Переменная окружения

```bash
# Разово при запуске
CLAUDE_CODE_TASK_LIST_ID=my-project claude

# Или глобально в .bashrc / .zshrc
export CLAUDE_CODE_TASK_LIST_ID="my-default-project"
```

### Способ 2: settings.json (рекомендуется для проекта)

Создай файл `.claude/settings.json` в корне проекта:

```json
{
  "env": {
    "CLAUDE_CODE_TASK_LIST_ID": "my-project"
  }
}
```

**Преимущества settings.json:**
- Конфиг хранится в репозитории
- Автоматически применяется при запуске в этой папке
- Не нужно помнить переменные окружения
- Можно добавить в `.gitignore` если ID должен быть локальным

**Полный пример settings.json:**

```json
{
  "model": "claude-sonnet-4-20250514",
  "env": {
    "CLAUDE_CODE_TASK_LIST_ID": "my-project",
    "CLAUDE_CODE_ENABLE_TASKS": "true"
  },
  "permissions": {
    "allow": [],
    "deny": []
  }
}
```

### Отключение Tasks

```bash
CLAUDE_CODE_ENABLE_TASKS=false claude
```

Или в settings.json:

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TASKS": "false"
  }
}
```

## Когда использовать

**Используй Tasks:**
- Рефакторинг 10+ файлов
- Декомпозиция большой фичи
- Параллельная работа субагентов
- Работа растянута на несколько дней

**Не нужны для:**
- Быстрые правки
- Одноразовые скрипты
- Простые вопросы

## Ограничения

- Хранение в `~/.claude/`, не в репозитории
- Нет поиска по завершённым задачам
- Нет git-версионирования

## Пример workflow

```bash
# 1. Запускаем с ID проекта
CLAUDE_CODE_TASK_LIST_ID=feature-user-auth claude

# 2. Просим Claude декомпозировать задачу
> Разбей задачу "Добавить авторизацию пользователей" на подзадачи

# 3. Claude создаёт задачи через TaskCreate
# 4. Выполняет их последовательно, учитывая зависимости
# 5. Отмечает выполненные через TaskUpdate

# 6. Закрываем терминал, открываем позже
CLAUDE_CODE_TASK_LIST_ID=feature-user-auth claude
> Покажи статус задач
# Всё на месте
```

## Интеграция с Git Worktree

Git worktree позволяет иметь несколько рабочих директорий одного репозитория с разными ветками. В комбинации с Tasks получается мощный workflow: **1 worktree = 1 ветка = 1 список задач**.

### Базовая настройка

```bash
# Структура проекта
~/projects/
├── myapp/                    # основной репозиторий (main)
├── myapp-feature-auth/       # worktree для фичи auth
└── myapp-feature-payments/   # worktree для фичи payments
```

### Создание worktree с Tasks

```bash
# 1. Создаём worktree для новой фичи
cd ~/projects/myapp
git worktree add ../myapp-feature-auth -b feature/auth

# 2. Переходим в worktree
cd ../myapp-feature-auth

# 3. Создаём settings.json с уникальным TASK_LIST_ID
mkdir -p .claude
echo '{
  "env": {
    "CLAUDE_CODE_TASK_LIST_ID": "myapp-feature-auth"
  }
}' > .claude/settings.json

# 4. Запускаем Claude — ID подхватится автоматически
claude
```

### Автоматизация через скрипт

Создай скрипт `new-feature.sh`:

```bash
#!/bin/bash
# Использование: ./new-feature.sh auth

FEATURE_NAME=$1
PROJECT_NAME=$(basename $(git rev-parse --show-toplevel))
WORKTREE_PATH="../${PROJECT_NAME}-feature-${FEATURE_NAME}"

# Создаём worktree
git worktree add "$WORKTREE_PATH" -b "feature/${FEATURE_NAME}"

# Настраиваем Tasks
mkdir -p "${WORKTREE_PATH}/.claude"
cat > "${WORKTREE_PATH}/.claude/settings.json" << EOF
{
  "env": {
    "CLAUDE_CODE_TASK_LIST_ID": "${PROJECT_NAME}-feature-${FEATURE_NAME}"
  }
}
EOF

echo "Worktree создан: $WORKTREE_PATH"
echo "Task list ID: ${PROJECT_NAME}-feature-${FEATURE_NAME}"
echo "Запусти: cd $WORKTREE_PATH && claude"
```

### Автоматический ID по имени ветки

Добавь в `.bashrc` / `.zshrc`:

```bash
# Автоматически задаёт TASK_LIST_ID по имени ветки
claude-auto() {
  local branch=$(git symbolic-ref --short HEAD 2>/dev/null)
  local project=$(basename $(git rev-parse --show-toplevel 2>/dev/null))

  if [[ -n "$branch" && -n "$project" ]]; then
    CLAUDE_CODE_TASK_LIST_ID="${project}-${branch}" claude "$@"
  else
    claude "$@"
  fi
}

alias ca='claude-auto'
```

Теперь `ca` автоматически использует ID вида `myapp-feature/auth`.

### Workflow с worktrees

```bash
# Терминал 1: работаем над auth
cd ~/projects/myapp-feature-auth
claude
> Декомпозируй задачу авторизации

# Терминал 2: параллельно работаем над payments
cd ~/projects/myapp-feature-payments
claude
> Декомпозируй задачу платежей

# Каждый worktree имеет:
# - Свою ветку
# - Свой список задач
# - Изолированные изменения файлов
```

### Удаление worktree

```bash
# После мержа фичи
cd ~/projects/myapp
git worktree remove ../myapp-feature-auth

# Задачи остаются в ~/.claude/ — можно удалить вручную или оставить как архив
```

### Преимущества

| Git Worktree | Tasks |
|--------------|-------|
| Изоляция файлов | Изоляция списка задач |
| Своя ветка | Свой контекст работы |
| Легко переключаться | Состояние сохраняется |
| Параллельная работа | Несколько агентов |

## Советы

1. **Один ID = один логический проект/фича**. Не смешивай разные задачи в одном списке.

2. **Называй ID понятно**: `feature-auth`, `bugfix-login`, `refactor-api` — легко найти потом.

3. **Для параллельной работы** несколько терминалов с одним ID позволяют агентам координироваться.

4. **Проверяй статус** командой "покажи задачи" перед продолжением работы.

5. **Worktree + Tasks** — используй для больших фич, где нужна изоляция и кода, и задач.
