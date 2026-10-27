// babel.config.js — Babel configuration for GesturaMobile
// The worklets plugin is required by react-native-vision-camera and
// react-native-worklets-core to enable synchronous frame processor callbacks
// running on the native camera thread.
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      'react-native-worklets-core/plugin',
    ],
  };
};
