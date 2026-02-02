# Apple Integration Guide for Mimir

This document covers Apple-specific features that require an Apple Developer account ($99/year).

## Prerequisites

- Apple Developer Program membership: https://developer.apple.com/programs/
- Team ID (found in Account → Membership)
- Signing certificate configured in Xcode/Tauri

---

## 1. iCloud Keychain Password AutoFill

Allows users to save and autofill login credentials via iCloud Keychain.

### Status: Prepared, needs Apple Developer account

### Files Created:

1. **Entitlements file**: `src-tauri/Mimir.entitlements`
   - Contains `com.apple.developer.associated-domains` with `webcredentials:mimir-8cbb2.firebaseapp.com`

2. **Apple App Site Association**: `firebase/.well-known/apple-app-site-association`
   - Needs `TEAM_ID` replaced with actual Team ID

3. **Firebase Hosting config**: `firebase/firebase.json`
   - Serves AASA file with correct Content-Type

### Steps to Enable:

1. Get your Team ID from Apple Developer portal

2. Update `firebase/.well-known/apple-app-site-association`:
   ```json
   {
     "webcredentials": {
       "apps": ["YOUR_TEAM_ID.com.mimir.dictation"]
     }
   }
   ```

3. Deploy to Firebase Hosting:
   ```bash
   cd firebase
   npm install -g firebase-tools  # if not installed
   firebase login
   firebase init hosting  # select mimir-8cbb2 project
   firebase deploy --only hosting
   ```

4. Verify AASA is accessible:
   ```bash
   curl https://mimir-8cbb2.firebaseapp.com/.well-known/apple-app-site-association
   ```

5. Build signed app with entitlements:
   ```bash
   cargo tauri build
   ```

6. Test Password AutoFill in the signed .app

### Troubleshooting:

- AASA must be served over HTTPS
- Content-Type must be `application/json`
- Team ID must match signing certificate
- App must be signed (not dev build)

---

## 2. Sign in with Apple

Native Apple Sign-In for macOS users.

### Status: Phase 8 (T066-T073) - Backend commands exist as placeholders

### Implementation Required:

1. **Apple Developer Console**:
   - Register App ID with "Sign in with Apple" capability
   - Create Service ID for web authentication
   - Configure domains and redirect URLs

2. **Backend Configuration**:
   - Configure Apple Sign-In in mimir-server
   - Add Service ID and OAuth redirect URL

3. **Code Implementation** (`src/auth/apple.rs`):
   - Use `AuthenticationServices` framework via FFI
   - Native macOS sign-in flow
   - Exchange Apple credential for backend tokens

4. **Entitlements** (already in `Mimir.entitlements`):
   ```xml
   <key>com.apple.developer.applesignin</key>
   <array>
     <string>Default</string>
   </array>
   ```

### Files to Create/Modify:

- `src/auth/apple.rs` - Native Apple Sign-In bindings
- `src/auth/backend.rs` - Add `sign_in_with_apple()` method
- `src-tauri/src/auth_commands.rs` - Update `auth_login_apple` command

---

## 3. App Store Distribution

For distributing via Mac App Store.

### Requirements:

- Apple Developer Program membership
- App Store Connect account
- App-specific password for notarization
- Proper entitlements for sandbox

### Entitlements for App Store (`Mimir.entitlements`):

Already configured:
- `com.apple.security.app-sandbox` - Required for App Store
- `com.apple.security.network.client` - Network access
- `com.apple.security.device.audio-input` - Microphone
- `com.apple.security.files.user-selected.read-write` - File access

### Build for App Store:

```bash
# Set signing identity
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"
export APPLE_ID="your@email.com"
export APPLE_PASSWORD="app-specific-password"

# Build and sign
cargo tauri build --target universal-apple-darwin

# Notarize
xcrun notarytool submit target/release/bundle/macos/Mimir.app.zip \
  --apple-id $APPLE_ID \
  --password $APPLE_PASSWORD \
  --team-id TEAM_ID \
  --wait
```

---

## 4. Universal Purchase (iOS + macOS)

If planning iOS version later.

### Considerations:

- Same App ID for both platforms
- Shared iCloud container for sync
- Universal purchase in App Store Connect

---

## Quick Reference

| Feature | Requires | Status |
|---------|----------|--------|
| Password AutoFill | Team ID + AASA deployment | Prepared |
| Sign in with Apple | Apple Developer + backend config | Phase 8 |
| App Store | Developer Program + Notarization | Future |
| Push Notifications | APNs certificate | Not planned |

---

## Environment Variables for Apple Features

```bash
# Add to your shell profile or CI/CD
export APPLE_TEAM_ID="XXXXXXXXXX"
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"
export APPLE_ID="your@email.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
```

---

Last updated: 2025-12-27
