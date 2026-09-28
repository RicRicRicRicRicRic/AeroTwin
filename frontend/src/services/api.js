/**
 * AeroTwin AI API service layer.
 *
 * The renderer is a plain web container: everything it needs from the Python
 * sidecar goes through this module (fetch only, no Node access), so UI state
 * stays cleanly separated from transport concerns. The base URL is resolved
 * from (in order) the `?apiBaseUrl=` query parameter injected by Electron, a
 * persisted override, or the default development port.
 */

const STORAGE_KEY = 'aerotwin.apiBaseUrl';
const DEFAULT_BASE_URL = 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT_MS = 30000;
const POLL_INTERVAL_MS = 1200;
const POLL_TIMEOUT_MS = 15 * 60 * 1000;

/** Statuses that mean "no longer running" for jobs, assessments, and reports. */
export const TERMINAL_STATUSES = ['completed', 'failed'];

export const JOB_TYPES = {
  frameExtraction: 'frame_extraction',
  materialSegmentation: 'material_segmentation',
  elementDetection: 'element_detection',
  crackMapping: 'crack_mapping',
};

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function readStoredBaseUrl() {
  try {
    const fromQuery = new URLSearchParams(window.location.search).get('apiBaseUrl');
    if (fromQuery) {
      window.localStorage.setItem(STORAGE_KEY, fromQuery);
      return fromQuery;
    }
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

let baseUrl = (readStoredBaseUrl() || DEFAULT_BASE_URL).replace(/\/+$/, '');

export function getBaseUrl() {
  return baseUrl;
}

export function setBaseUrl(next) {
  baseUrl = String(next || '').trim().replace(/\/+$/, '') || DEFAULT_BASE_URL;
  try {
    window.localStorage.setItem(STORAGE_KEY, baseUrl);
  } catch {
    /* storage unavailable — the in-memory value still applies */
  }
  return baseUrl;
}

async function request(path, { method = 'GET', body, timeoutMs = REQUEST_TIMEOUT_MS, signal } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  if (signal) {
    signal.addEventListener('abort', () => controller.abort(), { once: true });
  }
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = text;
      }
    }
    if (!response.ok) {
      const detail =
        payload && typeof payload === 'object' && 'detail' in payload ? payload.detail : payload;
      const message =
        typeof detail === 'string' && detail
          ? detail
          : `Request failed with HTTP ${response.status}`;
      throw new ApiError(message, { status: response.status, detail });
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error.name === 'AbortError') {
      throw new ApiError(`Request timed out after ${Math.round(timeoutMs / 1000)}s`, { status: 0 });
    }
    throw new ApiError(
      `Cannot reach the AeroTwin backend at ${baseUrl}. Start the sidecar or check the URL.`,
      { status: 0, detail: String(error && error.message ? error.message : error) }
    );
  } finally {
    clearTimeout(timer);
  }
}

/** Poll `poll()` until `isDone(value)` is true (or the timeout elapses). */
export async function waitFor(
  poll,
  { isDone, onUpdate, intervalMs = POLL_INTERVAL_MS, timeoutMs = POLL_TIMEOUT_MS } = {}
) {
  const startedAt = Date.now();
  for (;;) {
    const value = await poll();
    if (onUpdate) onUpdate(value);
    if (!isDone || isDone(value)) return value;
    if (Date.now() - startedAt > timeoutMs) {
      throw new ApiError(
        `Timed out after ${Math.round(timeoutMs / 1000)}s waiting for completion`
      );
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

function query(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') search.append(key, value);
  });
  const text = search.toString();
  return text ? `?${text}` : '';
}

/** Strip empty/undefined fields so optional API parameters stay absent. */
function compact(payload) {
  const result = {};
  Object.entries(payload).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') result[key] = value;
  });
  return result;
}

export const api = {
  // --- health ------------------------------------------------------------
  health: () => request('/api/health', { timeoutMs: 5000 }),

  // --- video import & frame extraction -----------------------------------
  listVideos: () => request('/api/processing/videos'),
  ingestVideo: (sourcePath) =>
    request('/api/processing/ingest-video', {
      method: 'POST',
      body: { source_path: sourcePath },
      timeoutMs: 5 * 60 * 1000,
    }),
  startFrameExtraction: ({ filename, frameStep, secondsInterval, imageFormat = 'jpg' }) =>
    request('/api/processing/extract-frames', {
      method: 'POST',
      body: compact({
        filename,
        frame_step: frameStep,
        seconds_interval: secondsInterval,
        image_format: imageFormat,
      }),
    }),

  // --- jobs ---------------------------------------------------------------
  listJobs: (limit = 25) => request(`/api/processing/jobs${query({ limit })}`),
  getJob: (jobId) => request(`/api/processing/jobs/${encodeURIComponent(jobId)}`),

  // --- AI analysis --------------------------------------------------------
  startMaterialSegmentation: ({ framesPath, tileSize }) =>
    request('/api/processing/segment-materials', {
      method: 'POST',
      body: compact({ frames_path: framesPath, tile_size: tileSize }),
    }),
  startElementDetection: ({ framesPath }) =>
    request('/api/processing/detect-elements', {
      method: 'POST',
      body: { frames_path: framesPath },
    }),
  startCrackMapping: ({ framesPath, crackThreshold }) =>
    request('/api/processing/map-cracks', {
      method: 'POST',
      body: compact({ frames_path: framesPath, crack_threshold: crackThreshold }),
    }),

  // --- building profiles & assessments ------------------------------------
  listProfiles: () => request('/api/assessment/profiles'),
  createProfile: (payload) =>
    request('/api/assessment/profiles', { method: 'POST', body: payload }),
  getProfile: (profileId) => request(`/api/assessment/profiles/${encodeURIComponent(profileId)}`),
  startAssessment: ({ profileId, crackJobId, elementJobId }) =>
    request('/api/assessment/calculate', {
      method: 'POST',
      body: compact({
        profile_id: profileId,
        crack_job_id: crackJobId,
        element_job_id: elementJobId,
      }),
    }),
  listAssessments: (limit = 25) => request(`/api/assessment/assessments${query({ limit })}`),
  getAssessment: (assessmentId) =>
    request(`/api/assessment/assessments/${encodeURIComponent(assessmentId)}`),

  // --- reports -------------------------------------------------------------
  generateReport: ({ assessmentId, reportFormat = 'both', includeVisualMaps = true }) =>
    request('/api/reports/generate', {
      method: 'POST',
      body: {
        assessment_id: assessmentId,
        report_format: reportFormat,
        include_visual_maps: includeVisualMaps,
      },
    }),
  listReports: ({ assessmentId, limit = 25 } = {}) =>
    request(`/api/reports${query({ assessment_id: assessmentId, limit })}`),
  getReport: (reportId) => request(`/api/reports/${encodeURIComponent(reportId)}`),

  // --- artifacts (masks, maps, and report files for the viewers) -----------
  listArtifacts: (category, path = '') =>
    request(`/api/processing/artifacts/listing${query({ category, path })}`),
  fetchArtifactJson: (category, path) =>
    request(`/api/processing/artifacts/file${query({ category, path })}`),
};

/** Absolute URL streaming one artifact file (usable as an <img src>). */
export function artifactUrl(category, relativePath) {
  return `${baseUrl}/api/processing/artifacts/file${query({ category, path: relativePath })}`;
}

/**
 * Derive the category-relative path of a job output directory, e.g.
 * `…/processed/crack_maps/smoke/run_x/job_ab12` → `smoke/run_x/job_ab12`.
 */
export function artifactRelativePath(outputDir, category) {
  if (!outputDir || !category) return '';
  const parts = String(outputDir).split(/[\\/]+/).filter(Boolean);
  const index = parts.lastIndexOf(category);
  return index === -1 ? '' : parts.slice(index + 1).join('/');
}

/** Resolve when a job finishes; rejects with the job's stored error on failure. */
export async function waitForJob(jobId, { onUpdate } = {}) {
  const job = await waitFor(() => api.getJob(jobId), {
    isDone: (value) => TERMINAL_STATUSES.includes(value.status),
    onUpdate,
  });
  if (job.status === 'failed') {
    throw new ApiError(job.error || `Job ${jobId} failed`, { status: 0, detail: job.error });
  }
  return job;
}

/** Resolve when an assessment finishes; rejects with its stored error. */
export async function waitForAssessment(assessmentId, { onUpdate } = {}) {
  const assessment = await waitFor(() => api.getAssessment(assessmentId), {
    isDone: (value) => TERMINAL_STATUSES.includes(value.status),
    onUpdate,
  });
  if (assessment.status === 'failed') {
    throw new ApiError(assessment.error || `Assessment ${assessmentId} failed`, {
      status: 0,
      detail: assessment.error,
    });
  }
  return assessment;
}

/** Resolve when report generation finishes; rejects with its stored error. */
export async function waitForReport(reportId, { onUpdate } = {}) {
  const report = await waitFor(() => api.getReport(reportId), {
    isDone: (value) => TERMINAL_STATUSES.includes(value.status),
    onUpdate,
  });
  if (report.status === 'failed') {
    throw new ApiError(report.error || `Report ${reportId} failed`, {
      status: 0,
      detail: report.error,
    });
  }
  return report;
}

export default api;

