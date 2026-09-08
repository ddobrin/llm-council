/**
 * Utilities for the LLM Council UI matching the Java implementation.
 */

const MODEL_NAME_MAP = {
  'gemini-3.6-flash': 'Gemini 3.6 Flash',
  'gemini-3.7-flash': 'Gemini 3.7 Flash',
  'gemini-3.8-flash': 'Gemini 3.8 Flash',
  'gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
  'gemini-2.5-pro': 'Gemini 2.5 Pro',
  'gemini-2.5-flash': 'Gemini 2.5 Flash',
  'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
  'claude-opus-5': 'Claude Opus 5',
  'claude-opus-4-8': 'Claude Opus 4.8',
  'claude-haiku-4-5': 'Claude Haiku 4.5',
  'claude-fable-5': 'Claude Fable 5',
};

const MODEL_COLOR_MAP = {
  'gemini-3.6-flash': '#d97706', // amber
  'gemini-3.7-flash': '#4285f4', // google blue
  'gemini-3.8-flash': '#1da1f2', // cyan
  'gemini-3.1-pro-preview': '#6366f1', // indigo
  'gemini-2.5-pro': '#4285f4',
  'gemini-2.5-flash': '#1da1f2',
  'gemini-2.5-flash-lite': '#d97706',
  'claude-opus-5': '#10a37f',
  'claude-fable-5': '#d97706',
};

const PALETTE = ['#4285f4', '#1da1f2', '#d97706', '#10a37f', '#8b5cf6', '#ec4899'];

/**
 * Format model ID into friendly human-readable name.
 */
export function formatModelName(modelId) {
  if (!modelId) return 'Unknown Model';
  // Strip provider prefix if present (e.g. google/gemini-2.5-pro)
  const base = modelId.includes('/') ? modelId.split('/')[1] : modelId;
  if (MODEL_NAME_MAP[base]) return MODEL_NAME_MAP[base];
  if (MODEL_NAME_MAP[modelId]) return MODEL_NAME_MAP[modelId];

  // Fallback: capitalize words and replace dashes with spaces
  return base
    .replace(/^([a-z])/, (m) => m.toUpperCase())
    .replace(/[-_]([a-z0-9])/g, (_, p1) => ` ${p1.toUpperCase()}`);
}

/**
 * Get distinct avatar color for a model.
 */
export function getAvatarColor(modelId, index = 0) {
  if (!modelId) return PALETTE[index % PALETTE.length];
  const base = modelId.includes('/') ? modelId.split('/')[1] : modelId;
  if (MODEL_COLOR_MAP[base]) return MODEL_COLOR_MAP[base];
  if (MODEL_COLOR_MAP[modelId]) return MODEL_COLOR_MAP[modelId];

  // Hash-based color selection from palette
  let hash = 0;
  for (let i = 0; i < base.length; i++) {
    hash = (hash << 5) - hash + base.charCodeAt(i);
    hash |= 0;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

/**
 * Get seat letter (0 -> 'A', 1 -> 'B', etc.).
 */
export function seatLetter(index) {
  return String.fromCharCode(65 + index);
}

/**
 * Computes label-to-model mapping client-side (e.g. "Response A" -> modelId).
 */
export function computeLabelToModel(responses, councilModels = []) {
  if (!responses || responses.length === 0) return {};

  // Sort responses to match council models order if known
  const sorted = [...responses].sort((a, b) => {
    const idxA = councilModels.indexOf(a.model);
    const idxB = councilModels.indexOf(b.model);
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    return 0;
  });

  const mapping = {};
  sorted.forEach((resp, i) => {
    mapping[`Response ${seatLetter(i)}`] = resp.model;
  });
  return mapping;
}

/**
 * Returns display label like "Model A" for a given model ID based on mapping.
 */
export function getModelLabel(modelId, labelToModel) {
  if (!modelId || !labelToModel) return undefined;
  const entry = Object.entries(labelToModel).find(([, id]) => id === modelId);
  return entry ? entry[0].replace('Response ', 'Model ') : undefined;
}

/**
 * Extracts raw evaluation text before "FINAL RANKING:".
 */
export function getEvaluationText(evaluation) {
  if (!evaluation) return '';
  const marker = evaluation.search(/FINAL RANKING[:\s]/i);
  if (marker === -1) return evaluation.trim();
  return evaluation.substring(0, marker).trim();
}

/**
 * De-anonymizes evaluation text by replacing "Response X" with bold model name.
 */
export function deAnonymizeText(text, labelToModel) {
  if (!text || !labelToModel) return text || '';

  let result = text;
  Object.entries(labelToModel).forEach(([label, model]) => {
    const friendlyName = formatModelName(model);
    result = result.replace(new RegExp(label, 'g'), `**${friendlyName}**`);
  });
  return result;
}
