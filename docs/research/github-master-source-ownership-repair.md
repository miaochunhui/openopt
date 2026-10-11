# Source Ownership Repair

The first main-integrated anonymous readback returned `source_query_failed`
before any network attempt. Its original receipt remains unchanged at
`github-master-readback/maintenance-live-adb-v1.json` under the external
release artifact root. With isolated HOME/global/system configuration, infra
cannot read the differently owned main repository: real Git returns 128 and
reports dubious ownership. The identical local query with an exact process
`safe.directory` returns the expected committed HEAD.

The production change is one local source-query argument:
`-c safe.directory=<resolved ROOT>`. It neither grants wildcard/parent trust
nor writes repository/global/system configuration. The remote anonymous
command, environment, URL/ref, credential clearing, clean-source/blob checks,
create-once receipts and error semantics are unchanged. No network query,
credentials, permissions, Git history or main-tree files are modified here.

New offline controls use real Git with its different-owner test switch.
They check rejection without trust and with a wrong path, exact-path success,
process-only/nonpersistent trust, dirty-source rejection before network,
hidden committed-blob drift rejection and unchanged remote command.
The original author test file remains byte-identical. The original eight
independent controls require Mendel's explicit source transport, not an
assertion change or an author claim of independent acceptance.

Run the unchanged original test reader against the new pinned verification
receipt to recheck source, collection/JUnit membership and counts. These CPU
controls do not establish an actual remote readback or three-host delivery.
