---
name: add-ui-panel
description: Add a new view or panel to the React Desktop UI.
---

# Add UI Panel Skill

To add a new view/panel to the desktop app:

## Step 1: Update State Definition

- Open `apps/desktop/src/store/engineStore.ts`.
- Add the new view to the `AppView` type: `export type AppView = "transcript" | "settings" | "history" | "new_panel";`.

## Step 2: Create the Component

- Create `apps/desktop/src/components/NewPanel.tsx`.
- Use functional components.
- Read necessary state via `useEngineStore()`.

## Step 3: Integrate into App

- Open `apps/desktop/src/App.tsx`.
- Add conditional rendering in the `<main className="app-content">` block.

```tsx
{
  activeView === "new_panel" && <NewPanel />;
}
```

## Step 4: Add Navigation

- Open `apps/desktop/src/components/Header.tsx`.
- Add a new button or icon to the `header-right` div to toggle the `activeView` to your new panel.

## Step 5: Styling

- Add any required layout styles to `apps/desktop/src/styles/global.css`.
- Rely entirely on existing CSS variables for colors, borders, and backgrounds to ensure dark mode works automatically.
