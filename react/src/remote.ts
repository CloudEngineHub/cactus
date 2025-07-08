let _cactusToken: string | null = null;

export function setCactusToken(token: string | null): void {
  _cactusToken = token;
}

export async function getVertexAIEmbedding(text: string): Promise<number[]> {
  if (_cactusToken === null) {
    throw new Error('CactusToken not set. Please call CactusLM.init with cactusToken parameter.');
  }

  const projectId = 'cactus-v1-452518';
  const location = 'us-central1';
  const modelId = 'text-embedding-005';
  
  const endpoint = `https://${location}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${location}/publishers/google/models/${modelId}:predict`;

  const headers = {
    'Authorization': `Bearer ${_cactusToken}`,
    'Content-Type': 'application/json',
  };

  const requestBody = {
    instances: [{ content: text }]
  };

  const response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(requestBody),
  });

  if (response.status === 401) {
    _cactusToken = null;
    throw new Error('Authentication failed. Please update your cactusToken.');
  } else if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  const responseBody = await response.json();
  
  if (responseBody.error) {
    throw new Error(`API Error: ${responseBody.error.message}`);
  }
  
  const predictions = responseBody.predictions;
  if (!predictions || predictions.length === 0) {
    throw new Error('No predictions in response');
  }
  
  const embeddings = predictions[0].embeddings;
  const values = embeddings.values;
  
  return values;
}

export async function getVertexAICompletion(
  textPrompt: string,
  imageData?: string,
  imagePath?: string,
  mimeType?: string,
): Promise<string> {
  if (_cactusToken === null) {
    throw new Error('CactusToken not set. Please call CactusVLM.init with cactusToken parameter.');
  }

  const projectId = 'cactus-v1-452518';
  const location = 'global';
  const modelId = 'gemini-2.5-flash-lite-preview-06-17';
  
  const endpoint = `https://aiplatform.googleapis.com/v1/projects/${projectId}/locations/${location}/publishers/google/models/${modelId}:generateContent`;

  const headers = {
    'Authorization': `Bearer ${_cactusToken}`,
    'Content-Type': 'application/json',
  };

  const parts: any[] = [];
  
  if (imageData) {
    const detectedMimeType = mimeType || 'image/jpeg';
    parts.push({
      inlineData: {
        mimeType: detectedMimeType,
        data: imageData
      }
    });
  } else if (imagePath) {
    const detectedMimeType = mimeType || detectMimeType(imagePath);
    const RNFS = require('react-native-fs');
    const base64Data = await RNFS.readFile(imagePath, 'base64');
    parts.push({
      inlineData: {
        mimeType: detectedMimeType,
        data: base64Data
      }
    });
  }
  
  parts.push({ text: textPrompt });

  const requestBody = {
    contents: {
      role: 'user',
      parts: parts,
    }
  };

  const response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(requestBody),
  });

  if (response.status === 401) {
    _cactusToken = null;
    throw new Error('Authentication failed. Please update your cactusToken.');
  } else if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  const responseBody = await response.json();
  
  if (Array.isArray(responseBody)) {
    throw new Error('Unexpected response format: received array instead of object');
  }
  
  if (responseBody.error) {
    throw new Error(`API Error: ${responseBody.error.message}`);
  }
  
  const candidates = responseBody.candidates;
  if (!candidates || candidates.length === 0) {
    throw new Error('No candidates in response');
  }
  
  const content = candidates[0].content;
  const responseParts = content.parts;
  if (!responseParts || responseParts.length === 0) {
    throw new Error('No parts in response');
  }
  
  return responseParts[0].text || '';
}

export async function getTextCompletion(prompt: string): Promise<string> {
  return getVertexAICompletion(prompt);
}

export async function getVisionCompletion(prompt: string, imagePath: string): Promise<string> {
  return getVertexAICompletion(prompt, undefined, imagePath);
}

export async function getVisionCompletionFromData(prompt: string, imageData: string, mimeType?: string): Promise<string> {
  return getVertexAICompletion(prompt, imageData, undefined, mimeType);
}

function detectMimeType(filePath: string): string {
  const extension = filePath.toLowerCase().split('.').pop();
  switch (extension) {
    case 'jpg':
    case 'jpeg':
      return 'image/jpeg';
    case 'png':
      return 'image/png';
    case 'gif':
      return 'image/gif';
    case 'webp':
      return 'image/webp';
    default:
      return 'image/jpeg';
  }
} 