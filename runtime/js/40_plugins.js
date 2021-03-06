// Copyright 2018-2021 the Deno authors. All rights reserved. MIT license.
"use strict";

((window) => {
  const core = window.Deno.core;

  function openPlugin(filename) {
    const rid = core.opSync("op_open_plugin", filename);
    core.syncOpsCache();
    return rid;
  }

  window.__bootstrap.plugins = {
    openPlugin,
  };
})(this);
