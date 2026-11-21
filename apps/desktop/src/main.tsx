import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/global.css";
import "./styles/transcript.css";
import "./styles/settings.css";
import "./styles/history-profiles.css";
import "./styles/modals.css";
import "./styles/overlay.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
