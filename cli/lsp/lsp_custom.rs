// Copyright 2018-2022 the Deno authors. All rights reserved. MIT license.

use deno_core::serde::Deserialize;
use deno_core::serde::Serialize;
use tower_lsp::lsp_types as lsp;

pub const CACHE_REQUEST: &str = "deno/cache";
pub const PERFORMANCE_REQUEST: &str = "deno/performance";
pub const RELOAD_IMPORT_REGISTRIES_REQUEST: &str =
  "deno/reloadImportRegistries";
pub const VIRTUAL_TEXT_DOCUMENT: &str = "deno/virtualTextDocument";

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheParams {
  /// The document currently open in the editor.  If there are no `uris`
  /// supplied, the referrer will be cached.
  pub referrer: lsp_types::TextDocumentIdentifier,
  /// Any documents that have been specifically asked to be cached via the
  /// command.
  pub uris: Vec<lsp_types::TextDocumentIdentifier>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RegistryStateNotificationParams {
  pub origin: String,
  pub suggestions: bool,
}

pub enum RegistryStateNotification {}

impl lsp_types::notification::Notification for RegistryStateNotification {
  type Params = RegistryStateNotificationParams;

  const METHOD: &'static str = "deno/registryState";
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VirtualTextDocumentParams {
  pub text_document: lsp_types::TextDocumentIdentifier,
}
