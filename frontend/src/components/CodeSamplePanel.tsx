import type { TranscriptionOptions } from "../types";
import { optionsToConfig } from "../types";

interface Props {
  options: TranscriptionOptions;
}

function highlightPython(code: string): JSX.Element {
  const keywords = /\b(import|from|as|def|class|if|else|elif|for|in|return|yield|with|try|except|finally|raise|assert|pass|break|continue|while|and|or|not|is|None|True|False)\b/g;
  const builtins = /\b(open|print|len|str|int|float|list|dict|set|tuple|range|enumerate|zip|map|filter|sorted|sum|max|min|abs|all|any)\b/g;

  return (
    <>
      {code.split("\n").map((line, i) => {
        // Comments
        if (line.trim().startsWith("#")) {
          return <span key={i} className="text-gray-500">{line}{"\n"}</span>;
        }

        // Split line into parts, preserving strings
        let result: JSX.Element[] = [];
        let key = 0;

        // Match strings first (preserve them)
        const stringPattern = /(["'])((?:\\.|(?!\1).)*?)\1/g;
        let lastIndex = 0;
        let match;

        while ((match = stringPattern.exec(line)) !== null) {
          // Process text before the string
          if (match.index > lastIndex) {
            const before = line.substring(lastIndex, match.index);
            result.push(...highlightNonString(before, key));
            key += 100;
          }
          // Add the string
          result.push(
            <span key={key++} className="text-orange-300">{match[0]}</span>
          );
          lastIndex = match.index + match[0].length;
        }

        // Process remaining text after last string
        if (lastIndex < line.length) {
          const after = line.substring(lastIndex);
          result.push(...highlightNonString(after, key));
        }

        return <span key={i}>{result}{"\n"}</span>;
      })}
    </>
  );

  function highlightNonString(text: string, baseKey: number): JSX.Element[] {
    const parts: JSX.Element[] = [];
    let current = text;
    let key = baseKey;

    // Highlight keywords
    const keywordMatches = [...current.matchAll(keywords)];
    const builtinMatches = [...current.matchAll(builtins)];
    const allMatches = [...keywordMatches, ...builtinMatches]
      .sort((a, b) => (a.index ?? 0) - (b.index ?? 0));

    let lastIdx = 0;
    for (const m of allMatches) {
      const idx = m.index ?? 0;
      if (idx > lastIdx) {
        parts.push(<span key={key++} className="text-gray-400">{current.substring(lastIdx, idx)}</span>);
      }
      const isKeyword = keywords.test(m[0]);
      keywords.lastIndex = 0;
      parts.push(
        <span key={key++} className={isKeyword ? "text-purple-400" : "text-cyan-400"}>
          {m[0]}
        </span>
      );
      lastIdx = idx + m[0].length;
    }

    if (lastIdx < current.length) {
      parts.push(<span key={key++} className="text-gray-400">{current.substring(lastIdx)}</span>);
    }

    return parts;
  }
}

export default function CodeSamplePanel({ options }: Props) {
  const url = `${window.location.origin}/v1/audio/transcriptions`;
  const config = optionsToConfig(options);
  const configJson = JSON.stringify(config, null, 2);
  const curlCmd = `curl -X POST \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "file=@audio.mp3" \\
  -F "config=<config.json" \\
  ${url}`;

  const pythonCode = `import requests, json

url = "${url}"
api_key = "YOUR_API_KEY"
config = json.load(open("config.json"))

response = requests.post(url,
    headers={"Authorization": f"Bearer {api_key}"},
    files={"file": open("audio.mp3", "rb")},
    data={"config": json.dumps(config)},
)
result = response.json()

for p in result.get("paragraphs", []):
    print(f"[{p['start']:.1f}s] {p.get('speaker', 'Unknown')}:")
    print(f"  {p['text']}")
    if p.get("entity_counts"):
        print(f"  Entities: {p['entity_counts']}")
    if p.get("sentiment"):
        print(f"  Sentiment: {p['sentiment']['label']}")
    print()`;

  return (
    <div className="p-4 space-y-4">
      {/* Config JSON */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
            config.json
          </div>
          <div className="text-[10px] text-gray-600">
            All settings in one file
          </div>
        </div>
        <div className="relative">
          <pre className="p-4 bg-surface-2 border border-border rounded-lg text-[11px] font-mono leading-relaxed text-gray-400 whitespace-pre overflow-x-auto max-h-[400px] overflow-y-auto">
            {configJson.split("\n").map((line, i) => {
              // Syntax highlight: keys in blue, strings in orange, booleans/numbers in purple
              const highlighted = line
                .replace(/^(\s*)"([^"]+)":/g, '$1<key>"$2"</key>:')
                .replace(/:\s*"([^"]*)"(,?)$/g, ': <str>"$1"</str>$2')
                .replace(/:\s*(true|false)(,?)$/g, ': <bool>$1</bool>$2')
                .replace(/:\s*(\d+\.?\d*)(,?)$/g, ': <num>$1</num>$2');

              return (
                <span key={i}>
                  {highlighted.split(/(<key>|<\/key>|<str>|<\/str>|<bool>|<\/bool>|<num>|<\/num>)/).reduce((acc: JSX.Element[], part, j) => {
                    if (part === "<key>" || part === "</key>" || part === "<str>" || part === "</str>" ||
                        part === "<bool>" || part === "</bool>" || part === "<num>" || part === "</num>") {
                      return acc;
                    }
                    // Determine color based on surrounding tags
                    const before = highlighted.split(/(<key>|<\/key>|<str>|<\/str>|<bool>|<\/bool>|<num>|<\/num>)/);
                    let inTag = "";
                    let count = 0;
                    for (let k = 0; k < before.length && count <= j; k++) {
                      if (before[k] === "<key>") inTag = "key";
                      else if (before[k] === "<str>") inTag = "str";
                      else if (before[k] === "<bool>") inTag = "bool";
                      else if (before[k] === "<num>") inTag = "num";
                      else if (before[k].startsWith("</")) inTag = "";
                      if (before[k] !== "" || k === 0) count++;
                    }

                    const color = inTag === "key" ? "text-blue-400" :
                                  inTag === "str" ? "text-orange-300" :
                                  inTag === "bool" ? "text-purple-400" :
                                  inTag === "num" ? "text-purple-400" : "";

                    acc.push(
                      <span key={j} className={color}>{part}</span>
                    );
                    return acc;
                  }, [])}
                  {"\n"}
                </span>
              );
            })}
          </pre>
          <button
            className="absolute top-2 right-2 px-2 py-0.5 text-[11px] bg-surface-3 border border-border rounded text-gray-500 hover:text-white hover:border-mvp-blue transition-all"
            onClick={() => navigator.clipboard.writeText(configJson)}
          >
            Copy
          </button>
        </div>
      </div>

      {/* curl */}
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 mb-2">
          cURL
        </div>
        <div className="relative">
          <pre className="p-4 bg-surface-2 border border-border rounded-lg text-[11px] font-mono leading-relaxed whitespace-pre-wrap">
            <span className="text-green-400">curl</span>{" "}
            <span className="text-blue-400">-X POST</span>{" \\\n"}
            {"  "}<span className="text-blue-400">-H</span>{" "}
            <span className="text-orange-300">"Authorization: Bearer YOUR_API_KEY"</span>{" \\\n"}
            {"  "}<span className="text-blue-400">-F</span>{" "}
            <span className="text-orange-300">"file=@audio.mp3"</span>{" \\\n"}
            {"  "}<span className="text-blue-400">-F</span>{" "}
            <span className="text-orange-300">"config=&lt;config.json"</span>{" \\\n"}
            {"  "}<span className="text-mvp-blue-light">{url}</span>
          </pre>
          <button
            className="absolute top-2 right-2 px-2 py-0.5 text-[11px] bg-surface-3 border border-border rounded text-gray-500 hover:text-white hover:border-mvp-blue transition-all"
            onClick={() => navigator.clipboard.writeText(curlCmd)}
          >
            Copy
          </button>
        </div>
      </div>

      {/* Python */}
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 mb-2">
          Python
        </div>
        <div className="relative">
          <pre className="p-4 bg-surface-2 border border-border rounded-lg text-[11px] font-mono leading-relaxed whitespace-pre-wrap">
            {highlightPython(pythonCode)}
          </pre>
          <button
            className="absolute top-2 right-2 px-2 py-0.5 text-[11px] bg-surface-3 border border-border rounded text-gray-500 hover:text-white hover:border-mvp-blue transition-all"
            onClick={() => navigator.clipboard.writeText(pythonCode)}
          >
            Copy
          </button>
        </div>
      </div>
    </div>
  );
}
