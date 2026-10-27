// metro.config.js — Metro bundler configuration for GesturaMobile
// Customizes asset resolution so `.tflite` model files are bundled as binary
// assets rather than treated as JavaScript source modules.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// 1. Bundle `.tflite` files as binary assets (images, fonts, etc.)
//    This lets us `require('./path/to/model.tflite')` and have Metro resolve
//    it to a numeric asset ID that react-native-fast-tflite can load.
if (!config.resolver.assetExts.includes('tflite')) {
  config.resolver.assetExts.push('tflite');
}

// 2. Prevent Metro from trying to parse `.tflite` as JS source.
//    Some Metro versions auto-include all extensions in sourceExts; stripping
//    it avoids "Unexpected token" parse errors on the binary blob.
config.resolver.sourceExts = config.resolver.sourceExts.filter(
  (ext) => ext !== 'tflite',
);

// 3. Also bundle `.task` files (MediaPipe hand-landmarker task file) if we
//    later decide to ship hand_landmarker.task inside the JS bundle.
if (!config.resolver.assetExts.includes('task')) {
  config.resolver.assetExts.push('task');
}

module.exports = config;