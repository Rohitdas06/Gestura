const { getDefaultConfig } = require('expo/metro-config');
const config = getDefaultConfig(__dirname);

// Whitelist the custom machine learning extension
config.resolver.assetExts.push('tflite');

module.exports = config;