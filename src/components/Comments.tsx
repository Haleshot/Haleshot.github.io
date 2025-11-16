import Giscus from "@giscus/react";
import { useEffect, useState } from "react";

interface CommentsProps {
  repo: `${string}/${string}`;
  repoId: string;
  category: string;
  categoryId: string;
  mapping: "pathname" | "url" | "title" | "og:title" | "specific" | "number";
  strict?: "0" | "1";
  reactionsEnabled?: "0" | "1";
  emitMetadata?: "0" | "1";
  inputPosition?: "top" | "bottom";
  lang?: string;
  loading?: "lazy" | "eager";
}

export default function Comments(props: CommentsProps) {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    const currentTheme = localStorage.getItem("theme");
    const browserTheme = window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
    return currentTheme || browserTheme;
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = ({ matches }: MediaQueryListEvent) => {
      setTheme(matches ? "dark" : "light");
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  useEffect(() => {
    const themeButton = document.querySelector("#theme-btn");
    const handleClick = () => {
      setTheme((prevTheme) => (prevTheme === "dark" ? "light" : "dark"));
    };

    themeButton?.addEventListener("click", handleClick);
    return () => themeButton?.removeEventListener("click", handleClick);
  }, []);

  return (
    <div className="mt-8">
      <Giscus
        repo={props.repo}
        repoId={props.repoId}
        category={props.category}
        categoryId={props.categoryId}
        mapping={props.mapping}
        strict={props.strict}
        reactionsEnabled={props.reactionsEnabled}
        emitMetadata={props.emitMetadata}
        inputPosition={props.inputPosition}
        theme={theme}
        lang={props.lang}
        loading={props.loading}
      />
    </div>
  );
}
