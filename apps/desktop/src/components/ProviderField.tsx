import { useEffect, useRef, useState } from "react";
import { PRESET_INPUT_LANGUAGES, PRESET_TARGET_LANGUAGES } from "../store/providerCatalog";
import type { ManifestField, ServiceOptionValue } from "../store/types";

// Presentational inputs for one manifest field. Parsing, validation, and vault calls live in
// the store; these components only render values and report user input.

function describedBy(...ids: (string | false | null | undefined)[]): string | undefined {
  const present = ids.filter(Boolean);
  return present.length > 0 ? present.join(" ") : undefined;
}

function FieldLabel({ htmlFor, field }: { htmlFor: string; field: ManifestField }) {
  return (
    <label htmlFor={htmlFor}>
      {field.label}
      {field.required && <span className="field-required"> (required)</span>}
    </label>
  );
}

function hintText(field: ManifestField): string {
  const parts: string[] = [];
  if (field.description) parts.push(field.description);
  if (field.minimum !== null && field.maximum !== null) {
    parts.push(`Range ${field.minimum} to ${field.maximum}.`);
  } else if (field.minimum !== null) {
    parts.push(`Minimum ${field.minimum}.`);
  } else if (field.maximum !== null) {
    parts.push(`Maximum ${field.maximum}.`);
  }
  return parts.join(" ");
}

export interface ProviderFieldProps {
  idPrefix: string;
  field: ManifestField;
  value: ServiceOptionValue | undefined;
  // Raw text kept by the store while a numeric entry does not parse.
  draft?: string;
  error?: string | null;
  disabled: boolean;
  onChange: (raw: string | boolean) => void;
}

export function ProviderField({
  idPrefix,
  field,
  value,
  draft,
  error,
  disabled,
  onChange,
}: ProviderFieldProps) {
  const id = `${idPrefix}-${field.key}`;
  const hint = hintText(field);
  const hintId = hint ? `${id}-hint` : null;
  const errorId = error ? `${id}-error` : null;
  const aria = {
    "aria-describedby": describedBy(hintId, errorId),
    "aria-invalid": error ? true : undefined,
    "aria-required": field.required || undefined,
  };
  const defaultText = field.default === null ? "" : String(field.default);

  let control: JSX.Element;
  let row = false;
  if (field.fieldType === "boolean" && field.default !== null) {
    row = true;
    control = (
      <input
        id={id}
        type="checkbox"
        checked={typeof value === "boolean" ? value : field.default === true}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        {...aria}
      />
    );
  } else if (field.fieldType === "boolean") {
    control = (
      <select
        id={id}
        value={typeof value === "boolean" ? String(value) : ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        {...aria}
      >
        <option value="">Provider default</option>
        <option value="true">On</option>
        <option value="false">Off</option>
      </select>
    );
  } else if (field.fieldType === "choice") {
    const emptyLabel = defaultText
      ? `Default (${defaultText})`
      : field.required
        ? "Select..."
        : "Provider default";
    control = (
      <select
        id={id}
        value={typeof value === "string" ? value : ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        {...aria}
      >
        <option value="">{emptyLabel}</option>
        {field.choices.map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    );
  } else if (field.key === "target_lang" && field.fieldType === "string") {
    const currentVal = typeof value === "string" ? value : defaultText;
    const isPreset = PRESET_TARGET_LANGUAGES.some((l) => l.code === currentVal);
    control = (
      <select
        id={id}
        value={currentVal}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        {...aria}
      >
        {!isPreset && currentVal && <option value={currentVal}>{currentVal}</option>}
        {PRESET_TARGET_LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.label}
          </option>
        ))}
      </select>
    );
  } else {
    const numeric = field.fieldType === "integer" || field.fieldType === "number";
    const shown = draft ?? (value === null || value === undefined ? "" : String(value));
    const hasLangList =
      field.fieldType === "string" && (field.key === "language" || field.key === "source_lang");
    const listId = hasLangList ? `${id}-presets` : undefined;
    control = (
      <>
        <input
          id={id}
          type="text"
          list={listId}
          inputMode={field.fieldType === "integer" ? "numeric" : numeric ? "decimal" : undefined}
          value={shown}
          placeholder={defaultText || "Provider default"}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          spellCheck={false}
          {...aria}
        />
        {hasLangList && (
          <datalist id={listId}>
            {PRESET_INPUT_LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.label}
              </option>
            ))}
          </datalist>
        )}
      </>
    );
  }

  return (
    <div
      className={`settings-field provider-field${row ? " settings-field-row" : ""}${error ? " has-error" : ""}`}
    >
      <FieldLabel htmlFor={id} field={field} />
      {control}
      {hint && (
        <p id={hintId ?? undefined} className="field-hint">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId ?? undefined} className="field-error">
          {error}
        </p>
      )}
    </div>
  );
}

export interface SecretProviderFieldProps {
  idPrefix: string;
  field: ManifestField;
  configured: boolean;
  pending: boolean;
  error?: string | null;
  disabled: boolean;
  // Receives the typed value once; the input is cleared before this is called.
  onSave: (secret: string) => Promise<boolean>;
  onDelete: () => void;
}

// Write-only: the value lives in the DOM input until submission and never in React or store
// state. Nothing can be read back, so a configured slot shows status, Replace, and Delete.
export function SecretProviderField({
  idPrefix,
  field,
  configured,
  pending,
  error,
  disabled,
  onSave,
  onDelete,
}: SecretProviderFieldProps) {
  const id = `${idPrefix}-${field.key}`;
  const statusId = `${id}-status`;
  const hintId = field.description ? `${id}-hint` : null;
  const errorId = error ? `${id}-error` : null;
  const inputRef = useRef<HTMLInputElement>(null);
  const [replacing, setReplacing] = useState(false);
  const showInput = !configured || replacing;
  const locked = disabled || pending;

  useEffect(() => {
    if (replacing) inputRef.current?.focus();
  }, [replacing]);

  const submit = async () => {
    const input = inputRef.current;
    if (!input || locked) return;
    const secret = input.value;
    input.value = "";
    const saved = await onSave(secret);
    if (saved) setReplacing(false);
  };

  return (
    <div className={`settings-field provider-field secret-field${error ? " has-error" : ""}`}>
      <div className="secret-field-header">
        {showInput ? (
          <label htmlFor={id}>
            {field.label}
            {field.required && <span className="field-required"> (required)</span>}
          </label>
        ) : (
          <span className="secret-field-label">{field.label}</span>
        )}
        <span
          id={statusId}
          className={`secret-status ${configured ? "secret-status-configured" : "secret-status-missing"}`}
        >
          {pending ? "Saving..." : configured ? "Configured" : "Not configured"}
        </span>
      </div>

      {showInput && (
        <div className="settings-field-group">
          <input
            id={id}
            ref={inputRef}
            type="password"
            className="settings-api-key-input"
            autoComplete="new-password"
            spellCheck={false}
            placeholder={configured ? "Enter a replacement value" : "Enter a value"}
            disabled={locked}
            aria-required={field.required || undefined}
            aria-invalid={error ? true : undefined}
            aria-describedby={describedBy(statusId, hintId, errorId)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void submit();
              }
            }}
          />
          <button type="button" className="btn btn-secondary" onClick={() => void submit()} disabled={locked}>
            {configured ? "Save replacement" : "Save"}
          </button>
          {replacing && (
            <button type="button" className="btn-text" onClick={() => setReplacing(false)} disabled={pending}>
              Cancel
            </button>
          )}
        </div>
      )}

      {configured && !replacing && (
        <div className="secret-field-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setReplacing(true)}
            disabled={locked}
            aria-describedby={statusId}
          >
            Replace {field.label}
          </button>
          <button
            type="button"
            className="btn-text btn-text-danger"
            onClick={onDelete}
            disabled={locked}
            aria-describedby={statusId}
          >
            Delete {field.label}
          </button>
        </div>
      )}

      {hintId && (
        <p id={hintId} className="field-hint">
          {field.description}
        </p>
      )}
      {error && (
        <p id={errorId ?? undefined} className="field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
