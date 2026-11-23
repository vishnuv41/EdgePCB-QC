// API Client for EdgePCB-QC FastAPI Backend

export interface SystemHealth {
  status: string;
  station_id: string;
  station_label: string;
  application: string;
  mode: string;
  timestamp_utc: string;
  device: string;
  architecture: string;
  ai_acceleration: string;
  active_backend: string;
  hardware_accelerated: boolean;
  vision_model: string;
  active_preset: string;
  database: string;
  camera_status: string;
  camera_info: {
    status: string;
    camera_id: string;
    target_fps: number;
    measured_fps: number;
    is_streaming: boolean;
    resolution: string;
    last_error?: string;
  };
  hardware_audit?: Record<string, any>;
  advisory_engine?: {
    mode?: string;
    model_loaded?: boolean;
    model_name?: string;
    model_artifact?: string;
    quantization?: string;
    status_badge?: string;
    fallback_ready?: boolean;
    attribution?: string;
  };
}

export interface Telemetry {
  cpu_percent: number;
  cpu_thermal_c: number;
  throttling_status: string;
  ram_used_mb: number;
  ram_total_mb: number;
  ram_percent: number;
  disk_free_gb: number;
  disk_total_gb: number;
  disk_percent: number;
  timestamp?: string;
}

export interface Defect {
  defect_id: number;
  class: string;
  severity: 'CRITICAL' | 'MAJOR' | 'MINOR';
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  dimensions_px: [number, number]; // [width, height]
  crop_url?: string;
}

export interface PipelineTimings {
  capture_ms: number;
  quality_gate_ms: number;
  vision_inference_ms: number;
  decision_engine_ms: number;
  advisory_ms: number;
  total_latency_ms: number;
}

export interface InspectionAdvisory {
  what_detected: string;
  why_it_matters: string;
  operator_action: string;
  confidence_caveat: string;
}

export interface InspectionRecord {
  inspection_id: string;
  board_id: string;
  timestamp_utc: string;
  verdict: 'PASS' | 'INSPECT' | 'REJECT';
  initial_decision?: string;
  final_decision?: string;
  rule_id: string;
  rule_description: string;
  requires_electrical_test: boolean;
  raw_image_url: string;
  annotated_image_url: string;
  defects: Defect[];
  timings: PipelineTimings;
  advisory: InspectionAdvisory;
  image_dimensions?: [number, number];
  operator_override?: {
    overridden: boolean;
    original_verdict: string;
    new_verdict: string;
    supervisor_id: string;
    justification: string;
    timestamp: string;
  };
}

export interface SampleBoard {
  filename: string;
  title: string;
  board_id: string;
  description: string;
}

export interface AuditLogItem {
  timestamp: string;
  inspection_id: string;
  board: string;
  rule: string;
  final: string;
  defects: number;
  electrical: boolean;
  latency_ms: number;
  overridden: boolean;
  source: string;
}

export interface ReviewStats {
  total_reviewed: number;
  confirmed_defects: number;
  false_calls: number;
  uncertain: number;
  false_call_rate_percent: number;
  class_breakdown: Record<string, { reviewed: number; false_calls: number }>;
}

export interface DefectStatistics {
  total_inspections: number;
  decisions: {
    PASS: number;
    INSPECT: number;
    REJECT: number;
  };
  defect_frequencies: Record<string, number>;
}

export interface InferenceConfig {
  backend_mode: 'auto' | 'cpu' | 'hailo';
  active_backend: string;
  active_backend_label: string;
  hardware_accelerated: boolean;
  fallback_reason?: string;
  conf_threshold: number;
  iou_threshold: number;
  max_detections: number;
  default_conf?: number;
  default_iou?: number;
  default_max_detections?: number;
  performance_metrics?: {
    avg_inference_latency_ms: number;
    avg_total_latency_ms: number;
    avg_fps: number;
    total_inspections_tracked: number;
  };
}

export interface HardwareStatus {
  rpi_model: string;
  is_raspberry_pi: boolean;
  cpu_architecture: string;
  os_name: string;
  os_release: string;
  system_ram: {
    total_mb: number;
    used_mb: number;
    free_mb: number;
    percent_used: number;
  };
  cpu_thermal_c: number;
  throttling_status: string;
  hailo_device: {
    detected: boolean;
    status: string;
    device_id: string;
    device_type: string;
  };
  hailort_version: string;
  hef_model: {
    status: string;
    file_name: string;
    path: string;
    size_mb: number;
    sha256: string;
    expected_sha256: string;
    checksum_verified: boolean;
  };
  inference_pipeline: {
    configured_mode: string;
    active_backend: string;
    active_backend_label: string;
    fallback_reason: string;
    hardware_accelerated: boolean;
  };
  thresholds: {
    confidence: number;
    iou: number;
    max_detections: number;
  };
  performance_metrics: {
    avg_inference_latency_ms: number;
    avg_total_latency_ms: number;
    avg_fps: number;
    total_inspections_tracked: number;
  };
  backend_status: string;
}

export interface PipelineComparison {
  sample_tested: string;
  cpu_pipeline: {
    available: boolean;
    backend: string;
    model_name: string;
    inference_latency_ms: number;
    total_latency_ms: number;
    estimated_fps: number;
    defect_count: number;
    detections: Defect[];
  };
  hailo_pipeline: {
    available: boolean;
    backend: string;
    model_name: string;
    inference_latency_ms: number | null;
    total_latency_ms: number | null;
    estimated_fps: number;
    defect_count: number;
    detections: Defect[];
    error?: string;
  };
  prediction_agreement: {
    class_agreement_percent: number;
    mean_bounding_box_iou: number;
    matched_count: number;
    cpu_only_count: number;
    hailo_only_count: number;
    matches: Array<{
      cpu_class: string;
      hailo_class: string;
      class_match: boolean;
      iou: number;
      cpu_conf: number;
      hailo_conf: number;
    }>;
  };
}

const API_BASE = '/api/v1';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const errorText = await res.text().catch(() => '');
    throw new Error(`API error ${res.status} on ${url}: ${errorText || res.statusText}`);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<SystemHealth> {
    return fetchJson<SystemHealth>(`${API_BASE}/system/health`);
  },

  async getTelemetry(): Promise<Telemetry> {
    return fetchJson<Telemetry>(`${API_BASE}/hardware/telemetry`);
  },

  async getCameraStatus() {
    return fetchJson<any>(`${API_BASE}/camera/status`);
  },

  async startCamera(resolution = '1280x720', cameraId = 'usb:0') {
    return fetchJson<{ status: string; resolution: string; is_streaming: boolean }>(
      `${API_BASE}/camera/start`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution, camera_id: cameraId }),
      }
    );
  },

  async stopCamera() {
    return fetchJson<{ status: string; is_streaming: boolean }>(
      `${API_BASE}/camera/stop`,
      { method: 'POST' }
    );
  },

  getCameraFeedUrl(): string {
    return `${API_BASE}/camera/feed`;
  },

  async captureCamera(): Promise<InspectionRecord> {
    return fetchJson<InspectionRecord>(`${API_BASE}/camera/capture`, { method: 'POST' });
  },

  async getSamples(): Promise<{ samples: SampleBoard[] }> {
    return fetchJson<{ samples: SampleBoard[] }>(`${API_BASE}/samples`);
  },

  async inspectSample(filename: string): Promise<InspectionRecord> {
    return fetchJson<InspectionRecord>(`${API_BASE}/samples/${filename}/inspect`, { method: 'POST' });
  },

  async uploadAndInspect(file: File): Promise<InspectionRecord> {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJson<InspectionRecord>(`${API_BASE}/inspections/upload`, {
      method: 'POST',
      body: formData,
    });
  },

  async getInspection(id: string): Promise<InspectionRecord> {
    return fetchJson<InspectionRecord>(`${API_BASE}/inspections/${id}`);
  },

  async overrideInspection(id: string, newVerdict: string, justification: string, supervisorId = 'LEAD-INSPECTOR-01') {
    return fetchJson<{ status: string; new_verdict: string }>(
      `${API_BASE}/inspections/${id}/override`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          new_verdict: newVerdict,
          justification,
          supervisor_id: supervisorId,
        }),
      }
    );
  },

  async reviewInspection(
    id: string,
    reviewerDecision: 'confirmed_defect' | 'false_call' | 'uncertain',
    defectId = 1,
    defectClass = 'open_circuit',
    notes = '',
    reviewerId = 'OP-QC-LINE'
  ) {
    return fetchJson<{ status: string; reviewer_decision: string }>(
      `${API_BASE}/inspections/${id}/review`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviewer_decision: reviewerDecision,
          defect_id: defectId,
          defect_class: defectClass,
          notes,
          reviewer_id: reviewerId,
        }),
      }
    );
  },

  async getReviewStats(): Promise<ReviewStats> {
    return fetchJson<ReviewStats>(`${API_BASE}/reviews/stats`);
  },

  async getDefectStatistics(sourceFilter = 'real'): Promise<DefectStatistics> {
    return fetchJson<DefectStatistics>(`${API_BASE}/analytics/defect-statistics?source_filter=${sourceFilter}`);
  },

  async getAuditLogs(limit = 50, sourceFilter?: string): Promise<AuditLogItem[]> {
    const q = sourceFilter ? `?limit=${limit}&source_filter=${sourceFilter}` : `?limit=${limit}`;
    return fetchJson<AuditLogItem[]>(`${API_BASE}/audit/inspections${q}`);
  },

  getAuditCsvUrl(sourceFilter?: string): string {
    return sourceFilter ? `${API_BASE}/audit/export/csv?source_filter=${sourceFilter}` : `${API_BASE}/audit/export/csv`;
  },

  async getInferenceConfig(): Promise<InferenceConfig> {
    return fetchJson<InferenceConfig>(`${API_BASE}/inference/config`);
  },

  async updateInferenceConfig(config: {
    backend_mode?: string;
    conf_threshold?: number;
    iou_threshold?: number;
    max_detections?: number;
  }): Promise<InferenceConfig> {
    return fetchJson<InferenceConfig>(`${API_BASE}/inference/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
  },

  async resetInferenceThresholds(): Promise<InferenceConfig> {
    return fetchJson<InferenceConfig>(`${API_BASE}/inference/reset-thresholds`, {
      method: 'POST',
    });
  },

  async getHardwareStatus(): Promise<HardwareStatus> {
    return fetchJson<HardwareStatus>(`${API_BASE}/hardware/status`);
  },

  async comparePipelines(sampleFilename?: string): Promise<PipelineComparison> {
    return fetchJson<PipelineComparison>(`${API_BASE}/inference/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_filename: sampleFilename }),
    });
  },
};
