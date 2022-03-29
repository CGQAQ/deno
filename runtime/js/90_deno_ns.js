// Copyright 2018-2022 the Deno authors. All rights reserved. MIT license.
"use strict";

((window) => {
  const core = window.Deno.core;
  const __bootstrap = window.__bootstrap;
  __bootstrap.denoNs = {
    metrics: core.metrics,
    test: __bootstrap.testing.test,
    bench: __bootstrap.testing.bench,
    Process: __bootstrap.process.Process,
    run: __bootstrap.process.run,
    isatty: __bootstrap.tty.isatty,
    writeFileSync: __bootstrap.writeFile.writeFileSync,
    writeFile: __bootstrap.writeFile.writeFile,
    writeTextFileSync: __bootstrap.writeFile.writeTextFileSync,
    writeTextFile: __bootstrap.writeFile.writeTextFile,
    readTextFile: __bootstrap.readFile.readTextFile,
    readTextFileSync: __bootstrap.readFile.readTextFileSync,
    readFile: __bootstrap.readFile.readFile,
    readFileSync: __bootstrap.readFile.readFileSync,
    watchFs: __bootstrap.fsEvents.watchFs,
    chmodSync: __bootstrap.fs.chmodSync,
    chmod: __bootstrap.fs.chmod,
    chown: __bootstrap.fs.chown,
    chownSync: __bootstrap.fs.chownSync,
    copyFileSync: __bootstrap.fs.copyFileSync,
    cwd: __bootstrap.fs.cwd,
    makeTempDirSync: __bootstrap.fs.makeTempDirSync,
    makeTempDir: __bootstrap.fs.makeTempDir,
    makeTempFileSync: __bootstrap.fs.makeTempFileSync,
    makeTempFile: __bootstrap.fs.makeTempFile,
    memoryUsage: core.memoryUsage,
    mkdirSync: __bootstrap.fs.mkdirSync,
    mkdir: __bootstrap.fs.mkdir,
    chdir: __bootstrap.fs.chdir,
    copyFile: __bootstrap.fs.copyFile,
    readDirSync: __bootstrap.fs.readDirSync,
    readDir: __bootstrap.fs.readDir,
    readLinkSync: __bootstrap.fs.readLinkSync,
    readLink: __bootstrap.fs.readLink,
    realPathSync: __bootstrap.fs.realPathSync,
    realPath: __bootstrap.fs.realPath,
    removeSync: __bootstrap.fs.removeSync,
    remove: __bootstrap.fs.remove,
    renameSync: __bootstrap.fs.renameSync,
    rename: __bootstrap.fs.rename,
    version: __bootstrap.version.version,
    build: __bootstrap.build.build,
    statSync: __bootstrap.fs.statSync,
    lstatSync: __bootstrap.fs.lstatSync,
    stat: __bootstrap.fs.stat,
    lstat: __bootstrap.fs.lstat,
    truncateSync: __bootstrap.fs.truncateSync,
    truncate: __bootstrap.fs.truncate,
    ftruncateSync: __bootstrap.fs.ftruncateSync,
    ftruncate: __bootstrap.fs.ftruncate,
    errors: __bootstrap.errors.errors,
    // TODO(kt3k): Remove this export at v2
    // See https://github.com/denoland/deno/issues/9294
    customInspect: __bootstrap.console.customInspect,
    inspect: __bootstrap.console.inspect,
    env: __bootstrap.os.env,
    exit: __bootstrap.os.exit,
    execPath: __bootstrap.os.execPath,
    Buffer: __bootstrap.buffer.Buffer,
    readAll: __bootstrap.buffer.readAll,
    readAllSync: __bootstrap.buffer.readAllSync,
    writeAll: __bootstrap.buffer.writeAll,
    writeAllSync: __bootstrap.buffer.writeAllSync,
    copy: __bootstrap.io.copy,
    iter: __bootstrap.io.iter,
    iterSync: __bootstrap.io.iterSync,
    SeekMode: __bootstrap.io.SeekMode,
    read: __bootstrap.io.read,
    readSync: __bootstrap.io.readSync,
    write: __bootstrap.io.write,
    writeSync: __bootstrap.io.writeSync,
    File: __bootstrap.files.File,
    FsFile: __bootstrap.files.FsFile,
    open: __bootstrap.files.open,
    openSync: __bootstrap.files.openSync,
    create: __bootstrap.files.create,
    createSync: __bootstrap.files.createSync,
    stdin: __bootstrap.files.stdin,
    stdout: __bootstrap.files.stdout,
    stderr: __bootstrap.files.stderr,
    seek: __bootstrap.files.seek,
    seekSync: __bootstrap.files.seekSync,
    connect: __bootstrap.net.connect,
    listen: __bootstrap.net.listen,
    connectTls: __bootstrap.tls.connectTls,
    listenTls: __bootstrap.tls.listenTls,
    startTls: __bootstrap.tls.startTls,
    shutdown: __bootstrap.net.shutdown,
    fstatSync: __bootstrap.fs.fstatSync,
    fstat: __bootstrap.fs.fstat,
    fsyncSync: __bootstrap.fs.fsyncSync,
    fsync: __bootstrap.fs.fsync,
    fdatasyncSync: __bootstrap.fs.fdatasyncSync,
    fdatasync: __bootstrap.fs.fdatasync,
    symlink: __bootstrap.fs.symlink,
    symlinkSync: __bootstrap.fs.symlinkSync,
    link: __bootstrap.fs.link,
    linkSync: __bootstrap.fs.linkSync,
    permissions: __bootstrap.permissions.permissions,
    Permissions: __bootstrap.permissions.Permissions,
    PermissionStatus: __bootstrap.permissions.PermissionStatus,
    serveHttp: __bootstrap.http.serveHttp,
    resolveDns: __bootstrap.net.resolveDns,
    upgradeWebSocket: __bootstrap.http.upgradeWebSocket,
    upgradeHttp: __bootstrap.http.upgradeHttp,
    kill: __bootstrap.process.kill,
    addSignalListener: __bootstrap.signals.addSignalListener,
    removeSignalListener: __bootstrap.signals.removeSignalListener,
  };

  __bootstrap.denoNsUnstable = {
    emit: __bootstrap.compilerApi.emit,
    setRaw: __bootstrap.tty.setRaw,
    consoleSize: __bootstrap.tty.consoleSize,
    DiagnosticCategory: __bootstrap.diagnostics.DiagnosticCategory,
    loadavg: __bootstrap.os.loadavg,
    hostname: __bootstrap.os.hostname,
    osRelease: __bootstrap.os.osRelease,
    systemMemoryInfo: __bootstrap.os.systemMemoryInfo,
    networkInterfaces: __bootstrap.os.networkInterfaces,
    getUid: __bootstrap.os.getUid,
    applySourceMap: __bootstrap.errorStack.opApplySourceMap,
    formatDiagnostics: __bootstrap.errorStack.opFormatDiagnostics,
    sleepSync: __bootstrap.timers.sleepSync,
    listen: __bootstrap.netUnstable.listen,
    connect: __bootstrap.netUnstable.connect,
    listenDatagram: __bootstrap.netUnstable.listenDatagram,
    Listener: __bootstrap.netUnstable.Listener,
    umask: __bootstrap.fs.umask,
    futime: __bootstrap.fs.futime,
    futimeSync: __bootstrap.fs.futimeSync,
    utime: __bootstrap.fs.utime,
    utimeSync: __bootstrap.fs.utimeSync,
    HttpClient: __bootstrap.fetch.HttpClient,
    createHttpClient: __bootstrap.fetch.createHttpClient,
    http: __bootstrap.http,
    dlopen: __bootstrap.ffi.dlopen,
    UnsafePointer: __bootstrap.ffi.UnsafePointer,
    UnsafePointerView: __bootstrap.ffi.UnsafePointerView,
    UnsafeFnPointer: __bootstrap.ffi.UnsafeFnPointer,
    flock: __bootstrap.fs.flock,
    flockSync: __bootstrap.fs.flockSync,
    funlock: __bootstrap.fs.funlock,
    funlockSync: __bootstrap.fs.funlockSync,
    refTimer: __bootstrap.timers.refTimer,
    unrefTimer: __bootstrap.timers.unrefTimer,
  };
})(this);
