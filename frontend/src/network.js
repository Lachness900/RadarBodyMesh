/**
 * Build a backend origin on the same machine that served the dashboard.
 *
 * A phone opening http://192.168.1.20:5173 must contact 192.168.1.20 rather
 * than localhost, which would refer to the phone itself.
 */
export function backendOriginFromPage(pageUrl, protocol, port = 8000) {
  const backendUrl = new URL(pageUrl);
  backendUrl.protocol = protocol;
  backendUrl.port = String(port);
  backendUrl.pathname = "/";
  backendUrl.search = "";
  backendUrl.hash = "";
  return backendUrl.origin;
}
