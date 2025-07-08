import { initLlama, LlamaContext } from './index'
import type {
  ContextParams,
  CompletionParams,
  CactusOAICompatibleMessage,
  NativeCompletionResult,
  EmbeddingParams,
  NativeEmbeddingResult,
} from './index'
import { Telemetry } from './telemetry'
import { setCactusToken, getVertexAIEmbedding } from './remote'

interface CactusLMReturn {
  lm: CactusLM | null
  error: Error | null
}

export class CactusLM {
  private context: LlamaContext

  private constructor(context: LlamaContext) {
    this.context = context
  }

  static async init(
    params: ContextParams,
    onProgress?: (progress: number) => void,
    cactusToken?: string,
  ): Promise<CactusLMReturn> {
    if (cactusToken) {
      setCactusToken(cactusToken);
    }

    const configs = [
      params,
      { ...params, n_gpu_layers: 0 } 
    ];

    for (const config of configs) {
      try {
        const context = await initLlama(config, onProgress);
        return { lm: new CactusLM(context), error: null };
      } catch (e) {
        Telemetry.error(e as Error, {
          n_gpu_layers: config.n_gpu_layers ?? null,
          n_ctx: config.n_ctx ?? null,
          model: config.model ?? null,
        });
        if (configs.indexOf(config) === configs.length - 1) {
          return { lm: null, error: e as Error };
        }
      }
    }
    return { lm: null, error: new Error('Failed to initialize CactusLM') };
  }

  async completion(
    messages: CactusOAICompatibleMessage[],
    params: CompletionParams = {},
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> {
    return await this.context.completion({ messages, ...params }, callback);
  }

  async embedding(
    text: string,
    params?: EmbeddingParams,
    mode: string = 'local',
  ): Promise<NativeEmbeddingResult> {
    let result: NativeEmbeddingResult;
    let lastError: Error | null = null;

    if (mode === 'remote') {
      result = await this._handleRemoteEmbedding(text);
    } else if (mode === 'local') {
      result = await this._handleLocalEmbedding(text, params);
    } else if (mode === 'localfirst') {
      try {
        result = await this._handleLocalEmbedding(text, params);
      } catch (e) {
        lastError = e as Error;
        try {
          result = await this._handleRemoteEmbedding(text);
        } catch (remoteError) {
          throw lastError;
        }
      }
    } else if (mode === 'remotefirst') {
      try {
        result = await this._handleRemoteEmbedding(text);
      } catch (e) {
        lastError = e as Error;
        try {
          result = await this._handleLocalEmbedding(text, params);
        } catch (localError) {
          throw lastError;
        }
      }
    } else {
      throw new Error('Invalid mode: ' + mode + '. Must be "local", "remote", "localfirst", or "remotefirst"');
    }
    return result;
  }

  private async _handleLocalEmbedding(text: string, params?: EmbeddingParams): Promise<NativeEmbeddingResult> {
    return this.context.embedding(text, params);
  }

  private async _handleRemoteEmbedding(text: string): Promise<NativeEmbeddingResult> {
    const embeddingValues = await getVertexAIEmbedding(text);
    return {
      embedding: embeddingValues,
    };
  }

  async rewind(): Promise<void> {
    // @ts-ignore
    return this.context?.rewind()
  }

  async release(): Promise<void> {
    return this.context.release()
  }
} 