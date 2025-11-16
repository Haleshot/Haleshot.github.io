import Giscus, { type GiscusProps } from "@giscus/react";
import { useEffect, useState } from "react";

interface CommentsProps extends Omit<GiscusProps, "theme"> {
  lightTheme?: string;
  darkTheme?: string;
}

export default function Comments({
  lightTheme = "light",
  darkTheme = "dark",
  ...giscusConfig
}: CommentsProps) {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    const currentTheme = localStorage.getItem("theme");
    const browserTheme = window.matchMedia("(prefers-color-scheme: dark)")
      .matches
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
      setTheme(prevTheme => (prevTheme === "dark" ? "light" : "dark"));
    };

    themeButton?.addEventListener("click", handleClick);
    return () => themeButton?.removeEventListener("click", handleClick);
  }, []);

  return (
    <div className="mt-8">
      <Giscus
        {...giscusConfig}
        theme={theme === "light" ? lightTheme : darkTheme}
      />
    </div>
  );
}

