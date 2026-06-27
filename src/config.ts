export const SITE = {
  website: "https://www.tanmaypatil.com/", // replace this with your deployed domain
  author: "Tanmay Patil",
  profile: "https://www.tanmaypatil.com/",
  desc: "Machine Learning Engineer focused on GPU programming, inference optimization, and diffusion research. Blog posts, projects, and ML models by Tanmay Patil.",
  title: "Tanmay Patil's Blog",
  ogImage: "astropaper-og.jpg",
  ogLocale: "en_US", // Open Graph locale (og:locale)
  twitter: "@TanmayPatil79", // Twitter/X handle for twitter:site & twitter:creator
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: true,
    text: "Edit page",
    url: "https://github.com/Tanmaypatil123/Tanmaypatil123.github.io/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Bangkok", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
