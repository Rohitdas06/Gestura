// babel.config.js — Babel configuration for GesturaMobile
// babel-preset-expo already injects `react-native-worklets/plugin` last
// when that package is installed. Do not add worklets-core here: a second
// worklets plugin can crash Reanimated 4 on launch.
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
  };
};
