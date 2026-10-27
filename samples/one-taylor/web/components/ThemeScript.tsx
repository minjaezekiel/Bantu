/**
 * Applies the saved theme before first paint.
 *
 * This has to be a blocking inline script rather than a useEffect: an effect
 * runs after the first paint, so a user who chose dark sees a white flash on
 * every navigation. Reading localStorage is wrapped because it throws in a
 * private window with site data blocked, and a thrown error here would stop
 * the page rendering at all.
 */
export function ThemeScript() {
  const js = `(function(){try{var t=localStorage.getItem("ot-theme");
if(t==="dark"||t==="light"){document.documentElement.setAttribute("data-theme",t);}
}catch(e){}})();`;
  return <script dangerouslySetInnerHTML={{ __html: js }} />;
}
