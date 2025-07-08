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
  private initParams: ContextParams

  private constructor(context: LlamaContext, initParams: ContextParams) {
    this.context = context
    this.initParams = initParams
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
        return { lm: new CactusLM(context, config), error: null };
      } catch (e) {
        Telemetry.error(e as Error, config);
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
    const startTime = Date.now();
    let firstTokenTime: number | null = null;
    
    const wrappedCallback = callback ? (data: any) => {
      if (firstTokenTime === null) firstTokenTime = Date.now();
      callback(data);
    } : undefined;

    const result = await this.context.completion({ messages, ...params }, wrappedCallback);
    
    Telemetry.track({
      event: 'completion',
      tok_per_sec: (result as any).timings?.predicted_per_second,
      toks_generated: (result as any).timings?.predicted_n,
      ttft: firstTokenTime ? firstTokenTime - startTime : null,
    }, this.initParams);

    return result;
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

    Telemetry.track({
      event: 'embedding',
      mode: mode,
    }, this.initParams);

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