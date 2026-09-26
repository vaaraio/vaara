# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Linux OS layer: governs what an agent does to files, whatever the agent.

Adapters govern an agent through that agent's own hook API. This layer sits
under all of them, in the kernel, so an agent with no adapter is governed too.

- ``vaara run <agent>`` starts the agent confined by the ``vaara-agent``
  AppArmor profile and inside a cgroup of its own. Everything it starts
  inherits both.
- The floor (:mod:`vaara.oslayer.floor`): Vaara's trail, keys and settings,
  every harness's hook and settings files, and the files a login shell,
  systemd, cron and git run from. No process in the agent's tree changes
  them, including through sudo. The profile holds in the kernel, so it holds
  whether or not the guard is running.
- The guard (``vaara os-guard``, root): loads the profile, tags each launch,
  decides every open and exec an agent makes in the folders the operator
  picked in the app, and writes each decision and each floor refusal to its
  own trail.

Linux with AppArmor (Ubuntu, Debian, SUSE). Where AppArmor is not enabled,
``vaara run`` refuses to start the agent rather than start it unconfined.
"""
