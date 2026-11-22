import { useEffect, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import {
  MAX_PROFILE_CONTENT_BYTES,
  MAX_PROFILE_NAME_CHARS,
  type ContextProfile,
  type ContextProfileInfo,
  type ProfileContentFormat,
} from "../store/types";
import { utf8ByteLength } from "../store/conversationSlice";

export interface ContextProfilesPanelProps {
  onClose?: () => void;
}

export function ContextProfilesPanel({ onClose }: ContextProfilesPanelProps) {
  const {
    profiles,
    profilesLoading,
    profilesError,
    loadProfiles,
    getProfileDetail,
    createProfile,
    updateProfile,
    deleteProfile,
  } = useEngineStore();

  const [search, setSearch] = useState("");
  const [editingProfile, setEditingProfile] = useState<ContextProfile | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [draftContent, setDraftContent] = useState("");
  const [draftFormat, setDraftFormat] = useState<ProfileContentFormat>("markdown");
  const [deleteTarget, setDeleteTarget] = useState<ContextProfileInfo | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  useEffect(() => {
    void loadProfiles();
  }, [loadProfiles]);

  const filteredProfiles = profiles.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase()),
  );

  const currentContentBytes = utf8ByteLength(draftContent);
  const bytesRemaining = MAX_PROFILE_CONTENT_BYTES - currentContentBytes;

  const handleStartCreate = () => {
    setEditingProfile(null);
    setDraftName("");
    setDraftContent("");
    setDraftFormat("markdown");
    setValidationError(null);
    setIsCreating(true);
  };

  const handleStartEdit = async (info: ContextProfileInfo) => {
    setValidationError(null);
    setIsCreating(false);
    const detail = await getProfileDetail(info.profileId);
    if (detail) {
      setEditingProfile(detail);
      setDraftName(detail.name);
      setDraftContent(detail.content);
      setDraftFormat(detail.contentFormat);
    }
  };

  const handleCancelForm = () => {
    setIsCreating(false);
    setEditingProfile(null);
    setValidationError(null);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    const name = draftName.trim();
    if (!name) {
      setValidationError("Profile name is required.");
      return;
    }
    if (name.length > MAX_PROFILE_NAME_CHARS) {
      setValidationError(`Name must be ${MAX_PROFILE_NAME_CHARS} characters or less.`);
      return;
    }
    if (currentContentBytes > MAX_PROFILE_CONTENT_BYTES) {
      setValidationError(`Content exceeds 64 KiB limit (${currentContentBytes} bytes).`);
      return;
    }

    if (isCreating) {
      const created = await createProfile(name, draftContent, draftFormat);
      if (created) {
        setIsCreating(false);
      }
    } else if (editingProfile) {
      const updated = await updateProfile(
        editingProfile.profileId,
        name,
        draftContent,
        draftFormat,
        editingProfile.revision,
      );
      if (updated) {
        setEditingProfile(null);
      }
    }
  };

  const handleConfirmDelete = async () => {
    if (deleteTarget) {
      await deleteProfile(deleteTarget.profileId);
      setDeleteTarget(null);
      if (editingProfile?.profileId === deleteTarget.profileId) {
        setEditingProfile(null);
      }
    }
  };

  return (
    <div className="profiles-panel" role="region" aria-label="Context Profiles">
      <div className="profiles-header">
        <div>
          <h2 className="profiles-title">Context Profiles</h2>
          <p className="profiles-subtitle">
            Attach glossaries, domain terminology, and system instructions for AI assistance.
          </p>
        </div>
        <div className="profiles-header-actions">
          {!isCreating && !editingProfile && (
            <button className="btn btn-primary" onClick={handleStartCreate}>
              + New Profile
            </button>
          )}
          {onClose && (
            <button className="btn btn-secondary" onClick={onClose}>
              Close
            </button>
          )}
        </div>
      </div>

      {profilesError && (
        <div className="profiles-error-banner" role="alert">
          ⚠️ {profilesError}
        </div>
      )}

      {/* Editor Form Modal or In-line Editor */}
      {(isCreating || editingProfile) ? (
        <form className="profile-editor-form" onSubmit={handleSave}>
          <div className="profile-form-header">
            <h3>{isCreating ? "Create Context Profile" : `Edit Profile: ${editingProfile?.name}`}</h3>
            {editingProfile && (
              <span className="profile-revision-badge">Revision v{editingProfile.revision}</span>
            )}
          </div>

          {validationError && (
            <div className="profile-validation-error" role="alert">
              {validationError}
            </div>
          )}

          <div className="profile-form-group">
            <label className="profile-label">
              Profile Name
              <span className="profile-char-counter">
                {draftName.length} / {MAX_PROFILE_NAME_CHARS} chars
              </span>
            </label>
            <input
              type="text"
              className="profile-input"
              value={draftName}
              maxLength={MAX_PROFILE_NAME_CHARS}
              onChange={(e) => setDraftName(e.target.value)}
              placeholder="e.g. Healthcare Acronyms, Client Project Specs"
              required
              autoFocus
            />
          </div>

          <div className="profile-form-group">
            <label className="profile-label">
              Profile Content
              <span
                className={`profile-byte-counter ${bytesRemaining < 0 ? "counter-overflow" : ""}`}
              >
                {bytesRemaining >= 0 ? `${bytesRemaining} bytes remaining` : `Exceeded by ${-bytesRemaining} bytes`}
              </span>
            </label>
            <textarea
              className="profile-textarea"
              rows={10}
              value={draftContent}
              onChange={(e) => setDraftContent(e.target.value)}
              placeholder="Enter glossary definitions, background context, or instructions..."
            />
          </div>

          <div className="profile-form-actions">
            <button type="button" className="btn btn-secondary" onClick={handleCancelForm}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={bytesRemaining < 0}>
              {isCreating ? "Create Profile" : "Save Changes"}
            </button>
          </div>
        </form>
      ) : (
        <>
          {/* Toolbar */}
          <div className="profiles-toolbar">
            <input
              type="search"
              className="profiles-search-input"
              placeholder="Search profiles by name..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          {/* Profile Cards */}
          {profilesLoading ? (
            <div className="profiles-loading">Loading context profiles...</div>
          ) : filteredProfiles.length === 0 ? (
            <div className="profiles-empty">
              {profiles.length === 0
                ? "No context profiles yet. Create a profile to provide background terminology and instructions to AI assistance."
                : "No context profiles match your search."}
            </div>
          ) : (
            <div className="profiles-grid">
              {filteredProfiles.map((p) => (
                <div key={p.profileId} className="profile-card">
                  <div className="profile-card-header">
                    <h4 className="profile-card-title">{p.name}</h4>
                    <span className="profile-card-revision">v{p.revision}</span>
                  </div>
                  <div className="profile-card-meta">
                    <span className="profile-format-tag">{p.contentFormat.toUpperCase()}</span>
                    <span>{Math.round(p.contentBytes / 1024 * 10) / 10} KB</span>
                    <span>{new Date(p.updatedAtMs).toLocaleDateString()}</span>
                  </div>
                  <div className="profile-card-actions">
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleStartEdit(p)}
                      title="Edit Profile"
                    >
                      ✎ Edit
                    </button>
                    <button
                      className="btn btn-danger btn-sm"
                      onClick={() => setDeleteTarget(p)}
                      title="Delete Profile"
                    >
                      🗑 Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="profile-dialog-overlay" role="dialog" aria-modal="true">
          <div className="profile-dialog">
            <h3>Delete Context Profile?</h3>
            <p>
              Are you sure you want to delete <strong>"{deleteTarget.name}"</strong>?
            </p>
            <p className="profile-dialog-note">
              ℹ Deleting this profile removes it from future sessions. Past conversations that attached this profile retain their immutable frozen snapshot.
            </p>
            <div className="profile-dialog-actions">
              <button
                className="btn btn-secondary"
                onClick={() => setDeleteTarget(null)}
                autoFocus
              >
                Cancel
              </button>
              <button className="btn btn-danger" onClick={handleConfirmDelete}>
                Delete Profile
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
