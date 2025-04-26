import Foundation

public class CactusClient {
    // Make this private so it doesn't appear in the public interface
    private var ctx: CactusContextRef?
    
    public init() {
        ctx = cactus_context_create()
    }
    
    deinit {
        if let ctx = ctx {
            cactus_context_destroy(ctx)
        }
    }
    
    public func loadModel(modelPath: String) -> Bool {
        guard let ctx = ctx else { return false }
        return modelPath.withCString { cstr in
            cactus_context_load_model(ctx, cstr) == 1
        }
    }
} 