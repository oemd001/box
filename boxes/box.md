Below is the JavaScript (.js)-only version of the auth helper you asked for.
Drop each file into your Cypress project exactly as named.

⸻

1 ▸ cypress.config.js

// cypress.config.js  – Cypress v12 / v13 CommonJS syntax
const { defineConfig } = require('cypress');
const { execSync }     = require('child_process');
const fs               = require('fs');

function getAccessCookies () {
  /*  -------------------------------------------------------------
      1) Negotiate with PingFederate / Kerberos
      2) Store cookies in /tmp/cj.txt
      3) Pull out GSSSO + IBDLogin and return them
  ------------------------------------------------------------- */
  const QA_IDP_URL   = process.env.QA_IDP_URL;
  const CLIENT_ID    = process.env.CLIENT_ID;
  const REDIRECT_URI = process.env.REDIRECT_URI;

  const cmd = `curl -s -k -L --negotiate -u ":" \
    -b /tmp/cj.txt -c /tmp/cj.txt \
    "${QA_IDP_URL}?nonce=1234&scope=openid&response_type=id_token%20token&client_id=${CLIENT_ID}&access_token_manager_id=RefDefault&IdpAdapterId=Kerberos&response_mode=form_post&state=xyz&redirect_uri=${REDIRECT_URI}"`;

  execSync(cmd);

  const jar = fs.readFileSync('/tmp/cj.txt', 'utf8');
  const gssso    = /GSSSO\s+([^\s]+)/.exec(jar)?.[1];
  const ibdLogin = /IBDLogin\s+([^\s]+)/.exec(jar)?.[1];

  if (!gssso || !ibdLogin) {
    throw new Error('Kerberos handshake failed – cookies not found');
  }

  return { gssso, ibdLogin };
}

module.exports = defineConfig({
  e2e: {
    setupNodeEvents (on, config) {
      const { gssso, ibdLogin } = getAccessCookies();
      config.env.gssso    = gssso;
      config.env.ibdLogin = ibdLogin;
      return config;
    },
    baseUrl           : 'https://qa.ibdweb.site.gs.com',
    chromeWebSecurity : false
  }
});



⸻

2 ▸ cypress/support/data-utils.js

/// <reference types="cypress" />

Cypress.Commands.add('open_dealSetup', () => {
  const gssso    = Cypress.env('gssso');
  const ibdLogin = Cypress.env('ibdLogin');

  if (!gssso || !ibdLogin) {
    throw new Error('Missing gssso / ibdLogin cookies in Cypress.env');
  }

  cy.setCookie('GSSSO',    gssso);
  cy.setCookie('IBDLogin', ibdLogin);

  // visit only after cookies are in place
  cy.visit('/ecm-syndicate-link/#/deal/dealSetup/');
});



⸻

3 ▸ cypress/support/index.js

import './data-utils.js';

Cypress.Cookies.defaults({
  preserve: ['GSSSO', 'IBDLogin']
});

(If your Node version doesn’t allow import in CommonJS, flip that one line to
require('./data-utils.js');.)

⸻

4 ▸ cypress.env.json  (do not commit to Git)

{
  "QA_IDP_URL"  : "https://idfs-qa.gs.com/as/authorization.oauth2",
  "CLIENT_ID"   : "f04c2914dfc04b29a6a08e1683295bba",
  "REDIRECT_URI": "https://qa.ibdweb.site.gs.com/tokenLogin?authLevel=30000"
}

or export the same three variables in your shell before running Cypress.

⸻

5 ▸ Using the command in tests

// cypress/e2e/dealSetup.cy.js
it('Create Deal', () => {
  cy.open_dealSetup();     // ← replaces cy.visit()
  // …the rest of your test remains unchanged
});



⸻

Running with Chrome Beta

# Interactive GUI
npx cypress open --browser "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"

# Headless
npx cypress run  --browser "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta" --headless

(Use the Windows path if you’re on Windows, e.g.
"C:\\Program Files\\Google\\Chrome Beta\\Application\\chrome.exe".)

With these four JS files in place, Cypress grabs fresh Kerberos cookies, injects them, and your specs run authenticated—no more 401s.
