import {
  describeCommandError,
  forwardCommand,
  isTauriRuntime,
} from "../lib/engineBridge";
import type { EngineState } from "./engineStore";
import {
  MAX_LABEL_CHARS,
  MAX_PROFILES_PER_SESSION,
  MAX_PROFILE_CONTENT_BYTES,
  MAX_PROFILE_NAME_CHARS,
  MAX_TITLE_CHARS,
  type ContextProfile,
  type ContextProfileInfo,
  type ProfileContentFormat,
} from "./types";

const SESSION_PREFS_KEY = "echoflux:session_prefs";

interface SessionPrefs {
  sourceLabels: { mic: string; system: string };
  attachedProfileIds: string[];
}

function loadSessionPrefs(): SessionPrefs {
  const defaults: SessionPrefs = {
    sourceLabels: { mic: "Me", system: "Others" },
    attachedProfileIds: [],
  };
  if (typeof window === "undefined") return defaults;
  try {
    const raw = window.localStorage?.getItem(SESSION_PREFS_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw) as Partial<SessionPrefs>;
    const mic =
      typeof parsed?.sourceLabels?.mic === "string"
        ? parsed.sourceLabels.mic.slice(0, MAX_LABEL_CHARS)
        : "Me";
    const system =
      typeof parsed?.sourceLabels?.system === "string"
        ? parsed.sourceLabels.system.slice(0, MAX_LABEL_CHARS)
        : "Others";
    const attached = Array.isArray(parsed?.attachedProfileIds)
      ? parsed.attachedProfileIds
          .filter((id): id is string => typeof id === "string" && id.length > 0)
          .slice(0, MAX_PROFILES_PER_SESSION)
      : [];
    return {
      sourceLabels: { mic: mic || "Me", system: system || "Others" },
      attachedProfileIds: attached,
    };
  } catch {
    return defaults;
  }
}

function saveSessionPrefs(prefs: SessionPrefs): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage?.setItem(SESSION_PREFS_KEY, JSON.stringify(prefs));
  } catch {}
}

export function utf8ByteLength(str: string): number {
  return new TextEncoder().encode(str).length;
}

export interface ProfilesSliceFields {
  profiles: ContextProfileInfo[];
  activeProfile: ContextProfile | null;
  profilesLoading: boolean;
  profilesError: string | null;
  attachedProfileIds: string[];
  sourceLabels: { mic: string; system: string };
  isSetupOpen: boolean;
  pendingTitle: string;
  setupTitleDirty: boolean;

  loadProfiles: () => Promise<void>;
  getProfileDetail: (profileId: string) => Promise<ContextProfile | null>;
  createProfile: (
    name: string,
    content: string,
    contentFormat?: ProfileContentFormat,
  ) => Promise<ContextProfile | null>;
  updateProfile: (
    profileId: string,
    name: string,
    content: string,
    contentFormat?: ProfileContentFormat,
    expectedRevision?: number,
  ) => Promise<ContextProfile | null>;
  deleteProfile: (profileId: string) => Promise<boolean>;
  setAttachedProfileIds: (profileIds: string[]) => void;
  setSourceLabels: (labels: { mic: string; system: string }) => void;
  openSetup: () => void;
  closeSetup: () => void;
  setPendingTitle: (title: string, forNextSession?: boolean) => void;
  startSessionWithSetup: () => Promise<void>;
}

export function createProfileActions(
  set: (
    partial: Partial<EngineState> | ((state: EngineState) => Partial<EngineState>),
  ) => void,
  get: () => EngineState,
): ProfilesSliceFields {
  const initialPrefs = loadSessionPrefs();

  return {
    profiles: [],
    activeProfile: null,
    profilesLoading: false,
    profilesError: null,
    attachedProfileIds: initialPrefs.attachedProfileIds,
    sourceLabels: initialPrefs.sourceLabels,
    isSetupOpen: false,
    pendingTitle: "",
    setupTitleDirty: false,

    loadProfiles: async () => {
      set({ profilesLoading: true, profilesError: null });
      try {
        const res = await forwardCommand({ type: "profile_list" });
        const dataMsg = res.messages.find(
          (m) => (m as any).profiles !== undefined || m.type === "profile_list_result",
        );
        const list = (dataMsg as any)?.profiles || (dataMsg as any)?.data?.profiles || [];
        const profiles: ContextProfileInfo[] = list.map((p: any) => ({
          profileId: p.profile_id || p.profileId,
          name: p.name || "",
          contentFormat: p.content_format || p.contentFormat || "markdown",
          revision: p.revision ?? 1,
          contentBytes: p.content_bytes || p.contentBytes || 0,
          updatedAtMs: p.updated_at_ms || p.updatedAtMs || Date.now(),
        }));
        set({ profiles, profilesLoading: false });
      } catch (err: any) {
        set({
          profilesLoading: false,
          profilesError: describeCommandError(err?.code || "bridge_failed"),
        });
      }
    },

    getProfileDetail: async (profileId) => {
      try {
        const res = await forwardCommand({ type: "profile_get", profile_id: profileId });
        const dataMsg = res.messages.find((m) => (m as any).profile !== undefined);
        const p = (dataMsg as any)?.profile || (dataMsg as any)?.data?.profile;
        if (!p) return null;
        const profile: ContextProfile = {
          profileId: p.profile_id || p.profileId,
          name: p.name,
          content: p.content || "",
          contentFormat: p.content_format || p.contentFormat || "markdown",
          revision: p.revision ?? 1,
          createdAtMs: p.created_at_ms || p.createdAtMs || Date.now(),
          updatedAtMs: p.updated_at_ms || p.updatedAtMs || Date.now(),
        };
        set({ activeProfile: profile });
        return profile;
      } catch {
        return null;
      }
    },

    createProfile: async (name, content, contentFormat = "markdown") => {
      const trimmedName = name.trim().slice(0, MAX_PROFILE_NAME_CHARS);
      if (!trimmedName) return null;
      if (utf8ByteLength(content) > MAX_PROFILE_CONTENT_BYTES) {
        set({ profilesError: "Profile content exceeds 64 KiB limit" });
        return null;
      }
      try {
        const res = await forwardCommand({
          type: "profile_create",
          name: trimmedName,
          content,
          content_format: contentFormat,
        });
        const dataMsg = res.messages.find((m) => (m as any).profile !== undefined);
        const p = (dataMsg as any)?.profile || (dataMsg as any)?.data?.profile;
        await get().loadProfiles();
        if (p) {
          return {
            profileId: p.profile_id || p.profileId,
            name: p.name,
            content: p.content || content,
            contentFormat: p.content_format || p.contentFormat || contentFormat,
            revision: p.revision ?? 1,
            createdAtMs: p.created_at_ms || Date.now(),
            updatedAtMs: p.updated_at_ms || Date.now(),
          };
        }
        return null;
      } catch (err: any) {
        set({ profilesError: describeCommandError(err?.code || "bridge_failed") });
        return null;
      }
    },

    updateProfile: async (
      profileId,
      name,
      content,
      contentFormat = "markdown",
      expectedRevision,
    ) => {
      const trimmedName = name.trim().slice(0, MAX_PROFILE_NAME_CHARS);
      if (utf8ByteLength(content) > MAX_PROFILE_CONTENT_BYTES) {
        set({ profilesError: "Profile content exceeds 64 KiB limit" });
        return null;
      }
      try {
        const payload: Record<string, unknown> = {
          type: "profile_update",
          profile_id: profileId,
          name: trimmedName,
          content,
          content_format: contentFormat,
        };
        if (expectedRevision !== undefined) {
          payload.expected_revision = expectedRevision;
        }
        const res = await forwardCommand(payload as any);
        const dataMsg = res.messages.find((m) => (m as any).profile !== undefined);
        const p = (dataMsg as any)?.profile || (dataMsg as any)?.data?.profile;
        await get().loadProfiles();
        if (p) {
          return {
            profileId: p.profile_id || p.profileId,
            name: p.name,
            content: p.content || content,
            contentFormat: p.content_format || p.contentFormat || contentFormat,
            revision: p.revision ?? (expectedRevision ? expectedRevision + 1 : 1),
            createdAtMs: p.created_at_ms || Date.now(),
            updatedAtMs: p.updated_at_ms || Date.now(),
          };
        }
        return null;
      } catch (err: any) {
        set({ profilesError: describeCommandError(err?.code || "bridge_failed") });
        return null;
      }
    },

    deleteProfile: async (profileId) => {
      try {
        await forwardCommand({ type: "profile_delete", profile_id: profileId });
        set((state) => {
          const nextAttached = state.attachedProfileIds.filter((id) => id !== profileId);
          saveSessionPrefs({
            sourceLabels: state.sourceLabels,
            attachedProfileIds: nextAttached,
          });
          return {
            profiles: state.profiles.filter((p) => p.profileId !== profileId),
            attachedProfileIds: nextAttached,
            activeProfile:
              state.activeProfile?.profileId === profileId ? null : state.activeProfile,
          };
        });
        return true;
      } catch (err: any) {
        set({ profilesError: describeCommandError(err?.code || "bridge_failed") });
        return false;
      }
    },

    setAttachedProfileIds: (attachedProfileIds) => {
      const sliced = attachedProfileIds.slice(0, MAX_PROFILES_PER_SESSION);
      saveSessionPrefs({ sourceLabels: get().sourceLabels, attachedProfileIds: sliced });
      set({ attachedProfileIds: sliced });
    },

    setSourceLabels: (sourceLabels) => {
      const normalized = {
        mic: sourceLabels.mic.slice(0, MAX_LABEL_CHARS),
        system: sourceLabels.system.slice(0, MAX_LABEL_CHARS),
      };
      saveSessionPrefs({
        sourceLabels: normalized,
        attachedProfileIds: get().attachedProfileIds,
      });
      set({ sourceLabels: normalized });
    },

    openSetup: () => set({ isSetupOpen: true }),
    closeSetup: () => set({ isSetupOpen: false }),
    setPendingTitle: (title, forNextSession = false) =>
      set((state) => ({
        pendingTitle: title.slice(0, MAX_TITLE_CHARS),
        setupTitleDirty: forNextSession ? true : state.setupTitleDirty,
      })),

    startSessionWithSetup: async () => {
      const { pendingTitle, sourceLabels, attachedProfileIds } = get();
      const title = pendingTitle.trim() || "Session " + new Date().toLocaleString();
      try {
        const convRes = await forwardCommand({
          type: "conversation_create",
          title,
          source_labels: sourceLabels,
        });
        const convData = convRes.messages.find(
          (m) => (m as any).conversation_id || (m as any).conversation,
        );
        const convId =
          (convData as any)?.conversation_id || (convData as any)?.conversation?.conversation_id;
        if (convId && attachedProfileIds.length > 0) {
          await forwardCommand({
            type: "profile_attach",
            conversation_id: convId,
            profile_ids: attachedProfileIds,
          });
        }
        if (convId) {
          set({
            activeConversationId: convId,
            engineConversationId: convId,
            preparedConversationId: convId,
            isViewingHistory: false,
            isSetupOpen: false,
          });
        } else {
          set({ isViewingHistory: false, isSetupOpen: false });
        }
      } catch (err: any) {
        set({
          isSetupOpen: false,
          ...(isTauriRuntime()
            ? {
                appError: `Session setup failed: ${describeCommandError(err?.code || "bridge_failed")}`,
              }
            : {}),
        });
      }
      get().startPipeline();
    },
  };
}
