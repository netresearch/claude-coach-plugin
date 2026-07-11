## {{TITLE}}

**Trigger**: {{TRIGGER}}

**Action**: {{ACTION}}

**Verification**: {{VERIFICATION}}

{{#if EVIDENCE}}
**Evidence**:
{{#each EVIDENCE}}
- {{this.quote}} ({{this.timestamp}})
{{/each}}
{{/if}}
