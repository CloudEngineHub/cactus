Pod::Spec.new do |spec|
  spec.name          = "cactus"
  spec.version       = "0.2.0"
  spec.summary       = "Cactus native framework for iOS"
  spec.description   = "Native C++ framework for running AI models locally on iOS"
  spec.homepage      = "https://github.com/cactus-compute/cactus"
  spec.license       = { :type => "MIT" }
  spec.author        = { "Cactus" => "contact@cactus.ai" }
  
  spec.ios.deployment_target = "13.0"
  spec.swift_version = "5.0"
  
  # Distribute the XCFramework
  spec.vendored_frameworks = "cactus.xcframework"
  
  # System frameworks that cactus depends on
  spec.frameworks = "Accelerate", "Foundation", "Metal", "MetalKit"
  
  # Include the XCFramework in the source
  spec.source = { :path => "." }
  
  # Preserve the XCFramework path structure
  spec.preserve_paths = "cactus.xcframework"
  
  # Public header files (if needed)
  spec.public_header_files = "cactus.xcframework/*/cactus.framework/Headers/*.h"
end 