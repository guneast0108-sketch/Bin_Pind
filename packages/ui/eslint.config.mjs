import js from "@eslint/js";
import tseslint from "typescript-eslint";

// packages/ui = web/extension 공유 순수 React 컴포넌트.
// 컴포넌트가 늘어나면 eslint-plugin-react 추가 예정(Phase 4-3 첫 승격 시).
export default tseslint.config(
  { ignores: ["dist/**"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    rules: {
      // CLAUDE.md 절대 금지: packages/ui에 next/* 의존 추가.
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["next", "next/*"],
              message: "packages/ui는 next/* 의존 금지 (web/extension 공유 순수 컴포넌트).",
            },
          ],
        },
      ],
    },
  },
);
