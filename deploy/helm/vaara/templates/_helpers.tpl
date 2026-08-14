{{/*
SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
SPDX-License-Identifier: AGPL-3.0-or-later
*/}}

{{- define "vaara.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vaara.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "vaara.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "vaara.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: vaara
{{- end -}}

{{- define "vaara.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vaara.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "vaara.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "vaara.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "vaara.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}

{{/*
Configuration that cannot be caught by a schema, checked once so a bad install
fails at template time with an explanation instead of crash-looping in the
cluster with an argparse error in the logs.
*/}}
{{- define "vaara.validate" -}}
{{- if not (has .Values.proxy.mode (list "observe" "enforce")) -}}
{{- fail (printf "proxy.mode must be \"observe\" or \"enforce\", got %q" .Values.proxy.mode) -}}
{{- end -}}
{{- if eq .Values.proxy.mode "enforce" -}}
{{- if and (not .Values.proxy.allow) (not .Values.proxy.approvals.enabled) -}}
{{- fail "proxy.mode=enforce needs proxy.allow (tool-name globs) and/or proxy.approvals.enabled. With neither, every tool call is gated and clients appear to lose their tools. Nothing is damaged, but the session is unusable. Start with proxy.allow: ['mcp__*'] and tighten, or run observe first." -}}
{{- end -}}
{{- end -}}
{{- if .Values.signing.enabled -}}
{{- if not .Values.signing.existingSecret -}}
{{- fail "signing.enabled requires signing.existingSecret. This chart does not generate signing keys: a key minted in a template would rotate on every upgrade and every receipt signed by the old one would stop verifying. Run `vaara keygen`, create the Secret, and name it here." -}}
{{- end -}}
{{- end -}}
{{- if not .Values.persistence.enabled -}}
{{- if .Values.signing.enabled -}}
{{- fail "signing.enabled with persistence.enabled=false writes receipts to a container filesystem that is discarded on restart. Signed evidence that does not survive a pod restart is not evidence." -}}
{{- end -}}
{{- end -}}
{{- end -}}
