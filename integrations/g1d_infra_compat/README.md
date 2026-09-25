# G1D-infra compatibility layer

Compared with upstream `1018weijia/G1D-infra` (`ee5464a`):

- Pico mapping: right A = start/pause/resume; left Y = record; left X = stop and mark failed; right B = quit.
- Policy takeover: upstream uses rollback to a held endpoint, then A-key alignment and a blended handoff. The upstream pure helpers are vendored beside this layer for integration.
- The upstream `policy_deploy.py` and `run_deploy.sh` are included under `upstream/`; they remain opt-in because they require Unitree SDK, camera and policy-server settings that differ from the current dry-run bridge.
- Voice: upstream has no TTS implementation. `VoiceAnnouncer` adds an asynchronous optional system TTS backend and a sink callback. It announces recording, failure, policy, takeover, tracking-loss and quit events.

The layer accepts the existing project's normalized dictionaries, so it does not require the Unitree SDK or a wired connection. Attach `PicoEventMapper` to the current XR sample loop and call `VoiceAnnouncer.announce()` from the existing recorder/policy state transitions.
