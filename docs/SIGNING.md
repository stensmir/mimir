# macOS Code Signing для Mimir

Эта инструкция позволяет создать стабильную подпись приложения, чтобы разрешения Accessibility сохранялись между обновлениями.

## Проблема

При adhoc-подписи (`linker-signed`) Tauri генерирует identifier с хешем:
```
mimir_ui-d79e8666de2d105f
```

При каждой сборке хеш меняется → macOS TCC считает это новым приложением → сбрасываются разрешения.

## Решение: Self-Signed Certificate

### Шаг 1: Создать сертификат

1. Открыть **Keychain Access** (`/Applications/Utilities/Keychain Access.app`)
2. Menu → **Certificate Assistant** → **Create a Certificate...**
3. Настройки:
   - **Name:** `Mimir Development`
   - **Identity Type:** `Self Signed Root`
   - **Certificate Type:** `Code Signing`
4. Нажать **Create**

Сертификат появится в Keychain под именем "Mimir Development".

### Шаг 2: Проверить сертификат

```bash
security find-identity -v -p codesigning
```

Должен отобразиться `Mimir Development`.

## Сборка с подписью

### 1. Собрать приложение

```bash
cd src-tauri
cargo tauri build
```

### 2. Подписать

```bash
./scripts/sign-macos.sh
```

Или с кастомным путём:
```bash
./scripts/sign-macos.sh target/release/bundle/macos/Mimir.app
```

Или с другим сертификатом:
```bash
SIGNING_IDENTITY="My Custom Cert" ./scripts/sign-macos.sh
```

### 3. Установить

```bash
cp -R target/release/bundle/macos/Mimir.app /Applications/
```

## Проверка

```bash
codesign -dv /Applications/Mimir.app 2>&1 | grep Identifier
```

Ожидаемый результат:
```
Identifier=com.mimir.dictation
```

**Без хеша** — это значит подпись стабильная.

## Ограничения

- Self-signed сертификат работает **только локально** на машине разработчика
- Для распространения другим пользователям потребуется:
  - Apple Developer аккаунт ($99/год)
  - Нотаризация приложения
- При переносе на другую машину нужно создать новый сертификат

## Troubleshooting

### Сертификат не найден

```bash
security find-identity -v -p codesigning | grep "Mimir"
```

Если пусто — создайте сертификат по инструкции выше.

### Ошибка "errSecInternalComponent"

Keychain требует разблокировки:
```bash
security unlock-keychain ~/Library/Keychains/login.keychain-db
```

### Gatekeeper блокирует запуск

Для self-signed приложения при первом запуске:
1. Правый клик на приложении → **Open**
2. Или: System Preferences → Security & Privacy → **Open Anyway**
