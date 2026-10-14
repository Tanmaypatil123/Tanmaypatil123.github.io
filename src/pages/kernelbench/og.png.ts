import type { APIRoute } from "astro";
import { generateOgImageForKernelBench } from "@/utils/generateOgImages";

export const GET: APIRoute = async () => {
  const buffer = await generateOgImageForKernelBench();
  return new Response(new Uint8Array(buffer), {
    headers: { "Content-Type": "image/png" },
  });
};
