//
//  SwiftExampleApp.swift
//  SwiftExample
//
//  Created by Henry Ndubuaku on 25/04/2025.
//

import SwiftUI

import CactusSwift

@main
struct SwiftExampleApp: App {
    @State private var modelLoaded = false
    var body: some Scene {
        WindowGroup {
            ContentView(modelLoaded: $modelLoaded)
                .onAppear {
                    let client = CactusClient()
                    client.loadModel(modelPath: "cactus/cactus-tests/llm.gguf")
                    modelLoaded = true
                    // You can store `client` in a @StateObject or elsewhere if you need to use it later
                }
        }
    }
}
