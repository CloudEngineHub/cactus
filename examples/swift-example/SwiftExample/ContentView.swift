//
//  ContentView.swift
//  SwiftExample
//
//  Created by Henry Ndubuaku on 25/04/2025.
//

import SwiftUI

struct ContentView: View {
    @Binding var modelLoaded: Bool
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            if modelLoaded {
                Text("Model has been loaded!")
            } else {
                Text("Loading model...")
            }
        }
        .padding()
    }
}

#Preview {
    ContentView(modelLoaded: .constant(false))
}
