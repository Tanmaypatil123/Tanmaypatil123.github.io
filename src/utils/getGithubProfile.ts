export interface GithubProfile {
  login: string;
  name: string | null;
  avatar_url: string;
  html_url: string;
  bio: string | null;
  public_repos: number;
  followers: number;
  following: number;
}

// Module-level cache so the profile is only fetched once per build,
// no matter how many times the component is rendered.
const cache = new Map<string, Promise<GithubProfile | null>>();

async function fetchProfile(username: string): Promise<GithubProfile | null> {
  try {
    const res = await fetch(`https://api.github.com/users/${username}`, {
      headers: { Accept: "application/vnd.github+json" },
    });
    if (!res.ok) return null;
    return (await res.json()) as GithubProfile;
  } catch {
    // Network failure / offline build — degrade gracefully.
    return null;
  }
}

export function getGithubProfile(
  username: string
): Promise<GithubProfile | null> {
  if (!cache.has(username)) {
    cache.set(username, fetchProfile(username));
  }
  return cache.get(username)!;
}
