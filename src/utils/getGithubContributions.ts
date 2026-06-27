export interface ContributionDay {
  date: string;
  count: number;
  level: 0 | 1 | 2 | 3 | 4;
}

export interface Contributions {
  total: number;
  days: ContributionDay[];
}

// Module-level cache so we only hit the API once per build.
const cache = new Map<string, Promise<Contributions | null>>();

async function fetchContributions(
  username: string
): Promise<Contributions | null> {
  try {
    // Public, no-auth API that returns the last year of contribution levels.
    const res = await fetch(
      `https://github-contributions-api.jogruber.de/v4/${username}?y=last`,
      { headers: { Accept: "application/json" } }
    );
    if (!res.ok) return null;
    const data = (await res.json()) as {
      total?: Record<string, number>;
      contributions?: ContributionDay[];
    };
    const days = data.contributions ?? [];
    if (days.length === 0) return null;
    const total =
      data.total?.lastYear ??
      days.reduce((sum, d) => sum + d.count, 0);
    return { total, days };
  } catch {
    return null;
  }
}

export function getGithubContributions(
  username: string
): Promise<Contributions | null> {
  if (!cache.has(username)) {
    cache.set(username, fetchContributions(username));
  }
  return cache.get(username)!;
}
