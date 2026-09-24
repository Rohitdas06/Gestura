// metro is known as the bundler for React Native. It is responsible for transforming and packaging your JavaScript code and assets into a format that can be run on mobile devices. The metro.config.js file is used to customize the behavior of the Metro bundler, such as specifying additional asset extensions or modifying the resolver settings.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add the extensions needed for TensorFlow.js models
config.resolver.assetExts.push('bin', 'txt', 'jpg', 'png', 'json');

module.exports = config;