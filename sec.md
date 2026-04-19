# Design

## 1) API0-R2

**Page Title**: API0-R2
 **ID**: API0_R2
 **Rule Name**: API authorization security protocol
 **Rule Summary**: OpenId Connect / OAuth 2.0 standards must be used
 **Reference**: [blank]
 **Creation Date**: 09 Dec 2022
 **Last reviewed date**: 16 May 2024

**Definition**:
 Basically, OpenID Connect / OAuth 2.0 standards must be used to request APIs.

Note: On browser-based applications (public Client), user session is managed on the client (browser): there is no session affinity mechanism and relevant session information is stored on the client and passed through APIs where needed. Nevertheless, if a user session needs to be maintained server-side and if there's no way to manage it partially or entirely at Business service level, then a frontend component (sometimes called “Backend for Frontend (BFF)”) in front of the API Provider platform is tolerated, as follows:

- User session management between the client and the frontend component is based on cookies.
- The frontend component requests APIs on the API Provider in compliance with OpenId Connect / OAuth 2.0 standards.
- All the credentials required for requesting APIs on the API Provider must be managed server-side (i.e. on frontend component): they must not transit on the client.

In a nutshell, any browser-based application that implements “Backend for Frontend (BFF)” pattern must be considered as a server-side application (confidential Client) with regards to APIs requests throughout the current document.

**Keywords**: [not visible in screenshot]
 **Review Status**: [not visible in screenshot]
 **Status comments**: [not visible in screenshot]

**中文翻译**
 **页面标题**：API0-R2
 **ID**：API0_R2
 **规则名称**：API 授权安全协议
 **规则摘要**：必须使用 OpenID Connect / OAuth 2.0 标准
 **参考**：[空白]
 **创建日期**：2022年12月09日
 **最近审查日期**：2024年05月16日

**定义**：
 原则上，请求 API 时必须使用 OpenID Connect / OAuth 2.0 标准。

说明：对于基于浏览器的应用（public Client），用户会话由客户端（浏览器）管理：不存在会话粘性机制，相关会话信息保存在客户端，并在需要时通过 API 传递。尽管如此，如果用户会话必须在服务端维护，并且无法在业务服务层面部分或全部实现，则允许在 API Provider 平台前增加一个前端组件（通常称为 “Backend for Frontend (BFF)”），条件如下：

- 客户端与前端组件之间的用户会话管理基于 cookie。
- 前端组件向 API Provider 请求 API 时，必须遵循 OpenID Connect / OAuth 2.0 标准。
- 向 API Provider 请求 API 所需的所有凭证都必须在服务端管理（即前端组件上），不得经过客户端。

简言之，任何实现了 “Backend for Frontend (BFF)” 模式的浏览器应用，在本文件关于 API 请求的上下文中，都应被视为服务端应用（confidential Client）。

**关键词**：[截图中不可见]
 **审查状态**：[截图中不可见]
 **状态说明**：[截图中不可见]

------

## 2) API0-R3

**Page Title**: API0-R3
 **ID**: API0_R3
 **Rule Name**: API security architecture
 **Rule Summary**: All requests on external-facing APIs must go through an API Provider as part of the exposition layer
 **Reference**: [blank]
 **Creation Date**: 09 Dec 2022
 **Last reviewed date**: 16 May 2024

**Definition**:
 The API Gateway must be part of the exposition layer.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API0-R3
 **ID**：API0_R3
 **规则名称**：API 安全架构
 **规则摘要**：所有对外暴露 API 的请求都必须经过作为暴露层组成部分的 API Provider
 **参考**：[空白]
 **创建日期**：2022年12月09日
 **最近审查日期**：2024年05月16日

**定义**：
 API Gateway 必须属于暴露层的一部分。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 3) API3-R1

**Page Title**: API3-R1
 **ID**: API3-R1
 **Rule Name**: Need to know principle
 **Rule Summary**: API design must be based on the need-to-know principle.
 **Reference**: [blank]
 **Creation Date**: 20 Mar 2015
 **Last reviewed date**: 20 Mar 2015

**Definition**:
 API design must be based on the need-to-know principle: only data which are necessary for the requester's duty are delivered to the requester.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API3-R1
 **ID**：API3-R1
 **规则名称**：按需知悉原则
 **规则摘要**：API 设计必须基于按需知悉原则。
 **参考**：[空白]
 **创建日期**：2015年03月20日
 **最近审查日期**：2015年03月20日

**定义**：
 API 设计必须遵循按需知悉原则：只向请求方提供其履行职责所必需的数据。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 4) API3-R4

**Page Title**: API3-R4
 **ID**: API3-R4
 **Rule Name**: Least-privilege principle
 **Rule Summary**: API design must be based on the least privilege principle (deny by default, allow on a case-by-case basis).
 **Reference**: [blank]
 **Creation Date**: 20 Mar 2015
 **Last reviewed date**: 20 Mar 2015

**Definition**:
 Least Privilege principle is a general rule that says all user, application and system functions should run with the least number of privileges necessary to complete their role.

As a consequence:

- It is forbidden to deliver a token which has access to “everything”.
- Only the requested resources must be accessible.
- Each Client must select only the grant flows necessary for its need.
- Avoid Origin '*' in CORS rules

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**:
 Rajout de la regle CORS :

1. Avoid Origin '*' in CORS rules

**中文翻译**
 **页面标题**：API3-R4
 **ID**：API3-R4
 **规则名称**：最小权限原则
 **规则摘要**：API 设计必须基于最小权限原则（默认拒绝，按个案放行）。
 **参考**：[空白]
 **创建日期**：2015年03月20日
 **最近审查日期**：2015年03月20日

**定义**：
 最小权限原则是一条通用规则，要求所有用户、应用和系统功能都只能使用完成其职责所需的最少权限。

因此：

- 禁止发放对“所有资源”都有访问权限的 token。
- 只能访问被请求的资源。
- 每个 Client 只能选择其实际需要的授权流程。
- 在 CORS 规则中应避免使用 Origin '*'

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：
 补充了 CORS 规则：

1. 在 CORS 规则中避免使用 Origin '*'

------

# Network

## 1) API0-R1

**ID**: API0_R1
 **Rule Name**: API network security protocol
 **Rule Summary**: TLS must be implemented everywhere
 **Reference**: CDF : TLS configuration
 **Creation Date**: 09 Dec 2022
 **Last reviewed date**: 16 May 2024

**Definition**:
 All flows must be TLS whatever the level of confidentiality of data is. This protects authentication credentials and data in transit against eavesdropping attacks. It also guarantees integrity of the transmitted data, authentication of services providers and allow Clients to authenticate against these services providers (like Authorization endpoint, Userinfo endpoint, Introspection endpoint...).

TLS implementation must comply with BNPP TLS and cryptographic requirements.

From Client application point of view:

- By default, server certification chain validation is to be used.
- For sensitive installed native applications or browser-based applications, server certificate pinning should be used.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **ID**：API0_R1
 **规则名称**：API 网络安全协议
 **规则摘要**：TLS 必须全面实施
 **参考**：CDF : TLS configuration
 **创建日期**：2022年12月09日
 **最近审查日期**：2024年05月16日

**定义**：
 所有通信流量都必须使用 TLS，无论数据保密级别如何。这样可以保护认证凭证和传输中的数据免受窃听攻击，也可以保障传输数据的完整性、服务提供方的身份认证，并允许客户端对这些服务提供方进行认证（例如 Authorization endpoint、Userinfo endpoint、Introspection endpoint 等）。

TLS 的实施必须符合 BNPP 的 TLS 和密码学要求。

从客户端应用的角度：

- 默认必须启用服务端证书链校验。
- 对于敏感的已安装原生应用或基于浏览器的应用，应使用服务端证书固定（certificate pinning）。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 2) API7-R3

**ID**: API7-R3
 **Rule Name**: Security HTTP headers
 **Rule Summary**: HTTP headers must be securely implemented.
 **Reference**: [blank]
 **Creation Date**: 20 Mar 2021
 **Last reviewed date**: 04 May 2022

**Definition**:

- The Business service provider must always send the “Content-Type” header to make sure the content of a given resources is interpreted correctly.
- The Business service provider should send “Cache-Control” and “Pragma” headers.
  - If “Cache-Control” isn’t specified, the API provider must set a “Cache-Control: no-store” header to prevent the response to be stored in any cache, and set "Pragma: no-cache" and "Expires: 0" for compliance with older browsers.
- The API provider must send an “X-Content-Type-Options: nosniff” header to make sure the browser does not try to detect a different Content-Type than what is actually sent (which could lead to cross site scripting (XSS) attacks).
- The API provider must remove any fingerprinting headers like X-Powered-By, Server, to prevent technical details and other valuable information from being sent back to attackers.

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**: [blank]

**中文翻译**
 **ID**：API7-R3
 **规则名称**：HTTP 安全头
 **规则摘要**：HTTP headers 必须安全地实现。
 **参考**：[空白]
 **创建日期**：2021年03月20日
 **最近审查日期**：2022年05月04日

**定义**：

- Business service provider 必须始终发送 “Content-Type” header，以确保资源内容被正确解析。
- Business service provider 应发送 “Cache-Control” 和 “Pragma” headers。
  - 如果没有指定 “Cache-Control”，则 API provider 必须设置 “Cache-Control: no-store”，防止响应被任何缓存存储，并设置 "Pragma: no-cache" 和 "Expires: 0" 以兼容旧浏览器。
- API provider 必须发送 “X-Content-Type-Options: nosniff” header，确保浏览器不会尝试推断一个与实际发送内容不同的 Content-Type（这可能导致跨站脚本 XSS 攻击）。
- API provider 必须移除任何指纹识别类 headers，例如 X-Powered-By、Server，以防止将技术细节和其他有价值信息返回给攻击者。

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：[空白]

------

# Token

## 1) API2-R2

**Page Title**: API2-R2
 **ID**: API2-R2
 **Rule Name**: Tokens lifespan
 **Rule Summary**: Tokens must be short-lived in accordance with the risk in case of leak.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 04 May 2022

**Definition**:
 Tokens could unintentionally be disclosed to an attacker via local log files, browser's history, referrer header, web server logs or other records kept by the Client application, or be stolen by a compromised Client application or a third-party script, or by eavesdropping. Then the attacker can try to inject them (with or without any modification) to act on behalf of the right owner. The token expiration time depends on the likelihood and impact of the risk associated with its leakage in accordance with the API's confidentiality and integrity requirements. If the Client application is untrusted, it cannot protect and store secrets for long-term use. Because of this, the user will have to reauthenticate and regrant access more often than with a trusted Client.

- By default:
  - Access Token: 900 seconds (15mn) maximum
  - ID Token: same expiration time than Access Token
  - JWT authentication token: 60 seconds maximum
  - Refresh Token: forbidden
  - Authorization code: 60 seconds (1mn) in one-time use
- API key:
  - API key: 2 years maximum
  - Access token: API key can not be use to obtain an access token => forbidden
- OpenId Connect Authorization Code Grant:
  - confidential client
    - Refresh Token: 90 days maximum (the shortest is the best)
  - public client
    - native application
      - Refresh Token: 90 days maximum in one-time use and store in biometrical container
    - other
      - Refresh Token: forbidden
      - If a renew of an access token is needed, a silent authentication must be performed: the presence of a SSO session cookie with the OpenID Provider is leveraged to request a new authorization code without showing any user interaction (such as authentication and consent) via a hidden iframe and the parameter “prompt=none” in the authentication request. Please note that silent authentication must not be automatically performed in background, but only performed when the user performs actions.
- OpenId Connect Device Code Grant:
  - native application
    - Refresh Token: 90 days maximum in one-time use and store in biometrical container
  - devicecode: 600 seconds maximum
  - Pooling: 5 minimum
- Token Exchange:
  - Access Token: 300 seconds (5mn) maximum

HTTP Basic Authent is forbidden.

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**:
 pour les SPA usage du refresh token possible :

- RT en rotation (preservation de la durée de vie)
- stockage dans un service worker

**中文翻译**
 **页面标题**：API2-R2
 **ID**：API2-R2
 **规则名称**：Token 生命周期
 **规则摘要**：Token 必须是短生命周期，并与泄露风险相匹配。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2022年05月04日

**定义**：
 Token 可能会无意中通过本地日志文件、浏览器历史记录、referrer header、Web 服务器日志或客户端应用保留的其他记录泄露给攻击者，也可能被受感染的客户端应用、第三方脚本或通过窃听而被窃取。攻击者随后可以尝试注入这些 token（无论是否修改），以代表合法所有者行事。Token 的过期时间应根据泄露风险发生的可能性及其影响来决定，并与 API 的保密性和完整性要求保持一致。如果客户端应用不可信，它就无法长期安全保存秘密。因此，与可信客户端相比，用户将需要更频繁地重新认证和重新授权。

- 默认情况下：
  - Access Token：最长 900 秒（15 分钟）
  - ID Token：与 Access Token 相同的过期时间
  - JWT authentication token：最长 60 秒
  - Refresh Token：禁止
  - Authorization code：最长 60 秒（1 分钟），且只能一次性使用
- API key：
  - API key：最长 2 年
  - Access token：不得使用 API key 去换取 access token，即禁止
- OpenID Connect Authorization Code Grant：
  - confidential client
    - Refresh Token：最长 90 天（越短越好）
  - public client
    - native application
      - Refresh Token：最长 90 天，一次性使用，并存放于生物识别保护容器中
    - 其他
      - Refresh Token：禁止
      - 如果需要续期 access token，必须执行 silent authentication：利用 OpenID Provider 上已有的 SSO session cookie，在不触发用户交互（例如登录或授权确认）的情况下，通过隐藏 iframe 和认证请求中的 “prompt=none” 参数来获取新的 authorization code。注意，silent authentication 不得在后台自动持续执行，只能在用户执行操作时触发。
- OpenID Connect Device Code Grant：
  - native application
    - Refresh Token：最长 90 天，一次性使用，并存放于生物识别保护容器中
  - devicecode：最长 600 秒
  - Polling：最少 5
- Token Exchange：
  - Access Token：最长 300 秒（5 分钟）

禁止使用 HTTP Basic Authent。

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：
 对于 SPA，允许使用 refresh token：

- 轮转 RT（保持生命周期控制）
- 存储在 service worker 中

------

## 2) API2-R3

**Page Title**: API2-R3
 **ID**: API2-R3
 **Rule Name**: Tokens format
 **Rule Summary**: Tokens format must ensure integrity and confidentiality in accordance with the context of use.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 04 May 2022

**Definition**:

API key:

- API keys must be unique, unpredictable (i.e. randomly generated with a strong generator algorithm to prevent prediction attacks and to guarantee uniqueness), complex, more than 128 bits long.

Access token:

- The main rules are:
  - this token must be unreadable for public clients on internet (opaque format ou JWE (encrypt for API Gateway))
  - This token must be unique, unpredictable (i.e. randomly generated with a strong generator algorithm to prevent prediction attacks and to guarantee uniqueness), complex, more than 128 bits long, not re-playable beyond its period of validity.
- By default, prefer to use opaque format
- In case of JWT format (JWS or JWE):
  - must be conform with RFC 7519 (JSON Web Token) and OIDC rules.
  - use algorithms compliant with BNPP
  - none signature is strictly forbidden (the alg parameter must be different than 'none')

Refresh token:

- This token must be unique, unpredictable (i.e. randomly generated with a strong generator algorithm to prevent prediction attacks and to guarantee uniqueness), complex, more than 128 bits long, not re-playable beyond its period of validity.
- opaque format

Authorization code:

- Authorization codes must be unique, unpredictable (i.e. randomly generated with a strong generator algorithm to prevent prediction attacks and guarantee uniqueness), complex, more than 128 bits long

JWT authentication token:

- JWT authentication tokens must be signed using an algorithm that is compliant with BNPP's approved cryptographic methods (for example, RS256).

ID token:

- JWT must signed by the OpenID provider using an algorithm that is compliant with BNPP's approved cryptographic methods (for example, RS256).
- Because the users data in ID tokens are only encoded, ID tokens must only contain standard claims.

User_code:

- User codes must be unpredictable, contain at least 8 characters in the following base character set "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ", blocked in case of 6 failed use attempts.

Device_code:

- must be unique, unpredictable, complex and more than 128 bits long.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API2-R3
 **ID**：API2-R3
 **规则名称**：Token 格式
 **规则摘要**：Token 的格式必须根据使用场景确保完整性和保密性。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2022年05月04日

**定义**：

API key：

- API key 必须唯一、不可预测（即使用强随机生成算法生成，以防预测攻击并保证唯一性）、复杂，且长度超过 128 bit。

Access token：

- 主要规则如下：
  - 对于互联网 public client，此 token 必须不可读（opaque 格式，或 JWE，即对 API Gateway 可解密的加密格式）
  - 此 token 必须唯一、不可预测（即使用强随机生成算法生成，以防预测攻击并保证唯一性）、复杂、长度超过 128 bit，并且在有效期之外不可被重放。
- 默认优先使用 opaque 格式
- 若使用 JWT 格式（JWS 或 JWE）：
  - 必须符合 RFC 7519（JSON Web Token）和 OIDC 规则
  - 必须使用符合 BNPP 要求的算法
  - 严禁使用 none 签名（alg 参数必须不同于 'none'）

Refresh token：

- 此 token 必须唯一、不可预测、复杂、长度超过 128 bit，并且在有效期之外不可被重放。
- 必须使用 opaque 格式

Authorization code：

- Authorization code 必须唯一、不可预测、复杂，且长度超过 128 bit。

JWT authentication token：

- JWT authentication token 必须使用符合 BNPP 批准密码学方法的算法签名（例如 RS256）。

ID token：

- JWT 必须由 OpenID provider 使用符合 BNPP 批准密码学方法的算法签名（例如 RS256）。
- 由于 ID token 中的用户数据只是被编码而非加密，因此 ID token 只能包含标准 claims。

User_code：

- User code 必须不可预测，至少包含 8 个字符，字符集限定为 "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"，且连续 6 次失败后必须被阻断。

Device_code：

- 必须唯一、不可预测、复杂，且长度超过 128 bit。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 3) API2-R4

**Page Title**: API2-R4
 **ID**: API2-R4
 **Rule Name**: SSO lifespan
 **Rule Summary**: SSO lifespan and access tokens lifespan must be synchronized.
 **Reference**: [blank]
 **Creation Date**: 21 Jun 2021
 **Last reviewed date**: 26 Oct 2021

**Definition**:
 SSO is the ability to authenticate once and never have to repeat the process for the duration of the session. SSO synchronization inherently creates a security risk: if an attacker gains access to user's credentials, he gains access to all the applications the user is authorized.

- The maximum SSO idle time must be 1800 seconds (30 mn).
- The maximum SSO lifespan must be 1 hours for customers, else 4 hours.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API2-R4
 **ID**：API2-R4
 **规则名称**：SSO 生命周期
 **规则摘要**：SSO 生命周期与 access token 生命周期必须同步。
 **参考**：[空白]
 **创建日期**：2021年06月21日
 **最近审查日期**：2021年10月26日

**定义**：
 SSO 指的是用户只认证一次，在整个会话期间无需重复认证。SSO 同步本身会带来安全风险：如果攻击者获取了用户凭证，那么他就可能访问用户有权访问的所有应用。

- SSO 最大空闲时间必须为 1800 秒（30 分钟）。
- SSO 最大生命周期：对客户为 1 小时，否则为 4 小时。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 4) API2-R5

**Page Title**: API2-R5
 **ID**: API2-R5
 **Rule Name**: Tokens management
 **Rule Summary**: Tokens must be managed (transport, storage, issuance, revocation) in accordance with their sensitivity.
 **Reference**: [blank]
 **Creation Date**: 21 Jun 2021
 **Last reviewed date**: 21 Jun 2021

**Definition**:

API key:

- API keys must not be passed via URL query parameters. They must be passed via the “Authorization” request header field, since doing so avoids logging by browsers or network components.

Access token:

- The Client application needs to keep access tokens safe.
- In case of a public client application, access tokens must not be stored but only stay in memory.
- Access tokens must not be passed via URL query parameters. They must be passed via the “Authorization” request header field, since doing so avoids logging by browsers or network elements.

Refresh token:

- The Client application needs to keep refresh tokens safe in a secure credential store.
  - For mobile applications, refresh tokens must be stored in the secure element of the device, the access of which is protected by biometrics.
    - Notice:
      - In case of 5 failed authentication attempts with biometrics, the access should be temporarily blocked 5 minutes. Then the application must delete the refresh token (and any other local data) automatically after 5 others failed attempts. Once deleted, the user needs to enroll again.
  - For server-side applications, refresh tokens must be stored encrypted.
  - For browser-based applications, refresh tokens are prohibited because there is no reliable way to ensure their confidentiality.
- Refresh tokens are one-time use: a new refresh token must be issued every time a new access token is issued, and the old one must be revoked. Refresh token rotation is intended to automatically detect and prevent attempts to use the same long-lived refresh token in parallel from different locations. This happens if a refresh token gets stolen from the Client application and is subsequently used by both the attacker and the legitimate user. The basic idea is to change the refresh token value with every refresh request in order to detect attempts to obtain access tokens using old refresh tokens. Since the OpenID Provider cannot determine whether the attacker or the legitimate user is trying to access, in case of such an access attempt all the refresh tokens and access tokens belonging to the user in the context of the Client application are revoked.
- Refresh tokens can be revoked at the initiative of the user (e.g. consent change, consent revocation) or at the initiative of BNPP Personal Finance (e.g. security issue, scope definition change). In this case, requests to get new access tokens from them will return an invalid grant error. A full authentication and grant of access must be performed.
- The number of active refresh tokens per Client application and user must be limited to 5. When the number runs into the limit, older refresh tokens must be revoked on server.

Authorization code:

- Authorization codes must not be stored. They are one-time use.
- Authorization codes should be tied to Client application (Client_id) which receive it.

JWT authentication token:

- JWT authentication tokens must not be stored. They are one-time use.

ID token:

- ID Tokens contain an “at_hash” (Access Token hash) property that must be used by the Client application to validate the Access Token.
- ID tokens may also contain the “acr” (authentication context class reference), “amr” (“authentication methods reference”) and “auth_time” claims which advertised the application about the current authentication strength, method and time when the user authentication occurred
- The ID Token passed by the authorization server to the client remains on the client and should never be broadcast. All informations in the ID token are reserved for the client for which it is intended. Access to the same data (and more) about the authenticated user from other processes must be performed through the userinfo endpoint.

Device code and user code:

- Devicecode and usercode must not be store
- usercode is one-time use.
- devicecode must not be transmit

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**:
 Modif:

- Correction sur Device_code et user_code. Inversion dans le doc de reference d'IT Risk

**中文翻译**
 **页面标题**：API2-R5
 **ID**：API2-R5
 **规则名称**：Token 管理
 **规则摘要**：Token 必须根据其敏感性进行管理（传输、存储、签发、撤销）。
 **参考**：[空白]
 **创建日期**：2021年06月21日
 **最近审查日期**：2021年06月21日

**定义**：

API key：

- API key 不得通过 URL query parameter 传递。必须通过 “Authorization” 请求头传递，因为这样可以避免被浏览器或网络组件记录。

Access token：

- 客户端应用必须妥善保护 access token。
- 对于 public client application，access token 不得持久化存储，只能保留在内存中。
- Access token 不得通过 URL query parameter 传递。必须通过 “Authorization” 请求头传递，以避免被浏览器或网络元素记录。

Refresh token：

- 客户端应用必须将 refresh token 安全存储在安全凭证存储中。
  - 对于移动应用，refresh token 必须存储在设备的 secure element 中，其访问受生物识别保护。
    - 说明：
      - 如果生物识别认证失败 5 次，应临时锁定访问 5 分钟。之后如果再失败 5 次，应用必须自动删除 refresh token（以及其他本地数据）。删除后，用户必须重新注册。
  - 对于服务端应用，refresh token 必须加密存储。
  - 对于基于浏览器的应用，禁止使用 refresh token，因为没有可靠方式确保其保密性。
- Refresh token 必须一次性使用：每次签发新的 access token 时，都必须同时签发新的 refresh token，并撤销旧的 refresh token。Refresh token 轮转的目的是自动检测并防止同一个长期有效 refresh token 被不同地点并行使用。如果 refresh token 从客户端应用中被窃取，攻击者和合法用户都使用它，就会发生这种情况。核心思想是每次 refresh 请求都更换 refresh token 的值，从而检测使用旧 refresh token 获取 access token 的行为。由于 OpenID Provider 无法区分当前请求者是攻击者还是合法用户，一旦发生这种访问尝试，属于该用户和该客户端应用上下文的所有 refresh token 和 access token 都必须被撤销。
- Refresh token 可以由用户主动撤销（例如 consent 变更、撤销 consent），也可以由 BNPP Personal Finance 主动撤销（例如安全问题、scope 定义变化）。在这种情况下，再用这些 refresh token 获取新 access token 的请求会返回 invalid grant 错误，必须重新执行完整认证和授权。
- 每个客户端应用和用户组合下的 active refresh token 数量必须限制为 5 个。当数量达到上限时，服务端必须撤销较旧的 refresh token。

Authorization code：

- Authorization code 不得存储，它只能一次性使用。
- Authorization code 应与接收它的客户端应用（Client_id）绑定。

JWT authentication token：

- JWT authentication token 不得存储，它只能一次性使用。

ID token：

- ID Token 包含 “at_hash”（Access Token 的哈希）属性，客户端应用必须使用它来校验 Access Token。
- ID token 还可以包含 “acr”（authentication context class reference）、“amr”（authentication methods reference）和 “auth_time” claims，用于告知应用当前认证强度、认证方式以及用户认证发生的时间。
- 授权服务器传递给客户端的 ID Token 必须保留在客户端，不应被广播。ID token 中的所有信息仅供其目标客户端使用。其他进程若要访问同一用户数据（或更多数据），必须通过 userinfo endpoint 获取。

Device code 和 user code：

- Devicecode 和 usercode 不得存储
- usercode 只能一次性使用
- devicecode 不得传输

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：
 修改：

- 修正 Device_code 和 user_code。IT Risk 参考文档中这两者写反了

------

# Client

## 1) API2-R1

**ID**: API2-R1
 **Rule Name**: Grant flow
 **Rule Summary**: The grant flow must be defined in accordance with the type of data and the Client application trust level.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 26 Oct 2021

**Definition**:
 The grant flow depends on the type of data and the Client application trust level. Here are the minimal requirements for each combination of the previous factors.

API access on behalf of a Client application (i.e. the Client application is the resource owner, that is to say user’s identity doesn’t matter):

- In case of a confidential client requesting public or private data:
  - Flow: OAuth JWT as Authorization Grant or Token Exchange
  - Consumer authentication: JWT authentication or Mutual TLS authentication.
- In case of an public Client requesting public data:
  - Flow: none
  - Consumer authentication: none
  - API key or HTTP Basic Authentication can be used as a non-authoritative tracking mechanism to provide basic operational metrics (such as relative load coming from different applications). It should not be the only basis for auditing that might lead to chargeback, simply because it is simply too easy to fake such elements on an untrusted Client.

API access on behalf of a user (i.e. a user is the resource owner and consents a right to the Client application):

- In case of native application requesting data:
  - Flow: Device Code Grant or Authorization Code Grant with Proof Key for Code Exchange (PKCE).
  - Consumer authentication: none
- In case of a confidential Client requesting data:
  - Flow: OpenId Connect Authorization Code Grant or Token Exchange
  - Consumer authentication: JWT authentication OR Mutual TLS authentication
- In case of a public Client requesting data:
  - Flow: OpenId Connect Authorization Code Grant with Proof Key for Code Exchange (PKCE).
  - Consumer authentication: none

**Keywords**: [blank]
 **Review Status**: [blank]
 **Status comments**: [blank]

**中文翻译**
 **ID**：API2-R1
 **规则名称**：授权流程
 **规则摘要**：授权流程必须根据数据类型和客户端应用的信任级别来确定。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年10月26日

**定义**：
 授权流程取决于数据类型以及客户端应用的信任级别。以下是不同组合情况下的最低要求。

代表客户端应用本身访问 API（即客户端应用是资源所有者，也就是说用户身份不重要）：

- 如果是 confidential client 请求 public 或 private 数据：
  - Flow：OAuth JWT as Authorization Grant 或 Token Exchange
  - Consumer authentication：JWT authentication 或 Mutual TLS authentication
- 如果是 public Client 请求 public 数据：
  - Flow：无
  - Consumer authentication：无
  - API key 或 HTTP Basic Authentication 可以作为一种非权威的跟踪机制，用于提供基础运营指标（例如不同应用带来的相对负载）。但它不应成为可能导致成本分摊审计的唯一依据，因为在不可信 Client 上伪造这类元素太容易了。

代表用户访问 API（即用户是资源所有者，并授权客户端应用代表自己访问）：

- 如果是 native application 请求数据：
  - Flow：Device Code Grant 或带 PKCE 的 Authorization Code Grant
  - Consumer authentication：无
- 如果是 confidential Client 请求数据：
  - Flow：OpenID Connect Authorization Code Grant 或 Token Exchange
  - Consumer authentication：JWT authentication 或 Mutual TLS authentication
- 如果是 public Client 请求数据：
  - Flow：带 PKCE 的 OpenID Connect Authorization Code Grant
  - Consumer authentication：无

**关键词**：[空白]
 **审查状态**：[空白]
 **状态说明**：[空白]

------

## 2) API2-R7

**ID**: API2-R7
 **Rule Name**: Client authentication
 **Rule Summary**: Client authentication must be based on certificates.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 09 Dec 2022

**Definition**:
 Client authentication only applies to confidential Clients.

The Client authentication must be performed either at application level thanks to JWT token authentication, or at network level thanks to Mutual TLS authentication. HTTP Basic or Digest Authentication methods are strictly forbidden.

JWT authentication allows application-to-application authentication. Mutual TLS authentication often ensures only a gateway-to-gateway authentication.

Conversly, Mutual TLS allows “sender constrained tokens”.

So, in case of internet-facing APIs whose confidentiality level is equal to 3 or higher, Mutual TLS authentication and “sender constrained tokens” should be implemented rather than JWT token authentication and bearer tokens.

Mutual TLS authentication :

- In accordance with TLS specifications, the API Provider must verify the full validity of the certificate presented by the Client application.
- The API Provider must verify that the subject of the certificate matches the Client application’s identifier (Client_id). The registration of this mapping must be performed during the technical setup and not in real-time.

JWT authentication :

- The JWT tokens must not be passed via URL query parameters. They must be passed via the “Authorization” request header field, since doing so avoids logging by browsers or network elements.
- The API Provider must verify the validity of the JWT token thanks to “exp” and “nbf” claims.
- The API Provider must verify the issuer (“iss” claim), subject (“sub” claim) and recipient (“aud” claim) to ensure the JWT token is used in the proper context. The token issuer (“iss” claim) and the subject (“sub” claim) must be the same. The subject (“sub” claim) must match the Client application’s identifier (Client_id).
- The API Provider must verify the signature of the JWT token. The public key must be transmitted during the technical setup and not in real-time. The API Provider should retrieve it thanks to its identifier (“kid” claim) and added to the header of the token. The API Provider must verify that the key identifier (“kid” claim) match the issuer (“iss” claim).

To enforce authentication and avoid identity theft, a client certificate from a CA dedicated to the authorization server should be implemented.

**Keywords**: [blank]
 **Review Status**: [blank]
 **Status comments**: [blank]

**中文翻译**
 **ID**：API2-R7
 **规则名称**：客户端认证
 **规则摘要**：客户端认证必须基于证书。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2022年12月09日

**定义**：
 客户端认证仅适用于 confidential Clients。

客户端认证必须通过以下两种方式之一完成：要么在应用层通过 JWT token authentication，要么在网络层通过 Mutual TLS authentication。严禁使用 HTTP Basic 或 Digest Authentication。

JWT authentication 支持应用到应用的认证。Mutual TLS authentication 往往只保证 gateway 到 gateway 的认证。

相反，Mutual TLS 支持 “sender constrained tokens”。

因此，对于面向互联网且保密级别大于等于 3 的 API，应优先实施 Mutual TLS authentication 和 “sender constrained tokens”，而不是 JWT token authentication 和 bearer tokens。

Mutual TLS authentication：

- 按照 TLS 规范，API Provider 必须完整校验客户端应用提交证书的有效性。
- API Provider 必须校验证书 subject 是否与客户端应用标识（Client_id）匹配。该映射关系必须在技术接入阶段注册，而不是实时建立。

JWT authentication：

- JWT token 不得通过 URL query parameter 传递，必须通过 “Authorization” 请求头传递，以避免被浏览器或网络元素记录。
- API Provider 必须通过 “exp” 和 “nbf” claims 校验 JWT token 的有效性。
- API Provider 必须校验 issuer（“iss” claim）、subject（“sub” claim）和 recipient（“aud” claim），以确保 JWT token 在正确的上下文中使用。Token issuer（“iss”）和 subject（“sub”）必须相同。Subject（“sub”）必须与客户端应用标识（Client_id）匹配。
- API Provider 必须校验 JWT token 的签名。公钥必须在技术接入阶段传递，而不是实时拉取。API Provider 应根据 token header 中的 key identifier（“kid” claim）来获取对应公钥，并且必须校验 “kid” 是否与 issuer（“iss” claim）匹配。

为了强化认证并避免身份盗用，应使用由专门面向授权服务器的 CA 签发的客户端证书。

**关键词**：[空白]
 **审查状态**：[空白]
 **状态说明**：[空白]

------

## 3) API2-R8

**ID**: API2-R8
 **Rule Name**: Session management
 **Rule Summary**: Connections must be securely managed (timeout, logout, unicity...).
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Any Client application that requires authenticated access must implement a user session mechanism that:

- Requires user re-authentication after a specified period of inactivity,
- Permits the user to close its current session at any time,
- Ensures all sessions are invalidated on the server side when the user logs out or when the user is automatically logged out, so that the existing session cannot be re-used,
- Enforces absolute session timeout,
- Limits the number of active concurrent sessions for a given user,
- Ensures that all other active sessions are terminated after a user successfully changes his password.

As a consequence:

- If a Client application is inactive for more than the access token lifespan (whatever it is in foreground or background), then the user will have to logon again.
- Access tokens must be explicitly deleted in the Client application and revoked on the OpenID Provider when users log out.
- When an application is uninstalled, then the refresh token must be explicitly deleted in the Client application and should be revoked on the OpenID Provider.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **ID**：API2-R8
 **规则名称**：会话管理
 **规则摘要**：连接必须被安全管理（超时、登出、唯一性等）。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 任何需要认证访问的客户端应用都必须实现一套用户会话机制，以确保：

- 用户在一段指定时间不活动后必须重新认证，
- 用户可以随时关闭当前会话，
- 当用户主动登出或被系统自动登出时，所有服务端会话都必须失效，从而保证现有会话不能被继续复用，
- 必须实施绝对会话超时，
- 必须限制同一用户同时活跃会话的数量，
- 用户成功修改密码后，所有其他活跃会话都必须被终止。

因此：

- 如果客户端应用的不活动时间超过 access token 生命周期（无论前台还是后台），用户都必须重新登录。
- 用户登出时，客户端应用必须显式删除 access token，并在 OpenID Provider 上将其撤销。
- 当应用被卸载时，refresh token 必须在客户端应用中被显式删除，并且应在 OpenID Provider 上被撤销。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 4) API2-R9

**ID**: API2-R9
 **Rule Name**: End-user Authentication
 **Rule Summary**: User authentication level depends on data confidentiality and integrity requirements
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 09 Dec 2022

**Definition**:

- If the confidentiality or the integrity level of one of the required APIs is equal to 3 or higher, then the minimum level of user’s authentication must be based on a multi-factor authentication method.
- Else if the confidentiality or the integrity level of one of the required APIs is equal to 2 or higher, then the minimum level of user’s authentication must be based on a mono-factor authentication method.
- Else no user’s authentication is required.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **ID**：API2-R9
 **规则名称**：终端用户认证
 **规则摘要**：用户认证强度取决于数据的保密性和完整性要求。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2022年12月09日

**定义**：

- 如果所需 API 中任意一个的保密性或完整性级别大于等于 3，则用户认证的最低要求必须是多因素认证。
- 否则，如果所需 API 中任意一个的保密性或完整性级别大于等于 2，则用户认证的最低要求必须是单因素认证。
- 否则，不要求用户认证。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 5) API2-R10

**ID**: API2_R10
 **Rule Name**: Client Identification
 **Rule Summary**: Each client identifier (client_id or APIKey) is unique.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Each client identifier (client_id or API_Key) is unique, represents the same component, and MUST NOT be used by another component.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **ID**：API2_R10
 **规则名称**：客户端标识
 **规则摘要**：每个客户端标识（client_id 或 APIKey）都必须唯一。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 每个客户端标识（client_id 或 API_Key）都必须唯一，代表同一个组件，且不得被其他组件复用。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 6) API6-R1

**ID**: API6-R1
 **Rule Name**: Replay attacks
 **Rule Summary**: Security measures must be implemented to prevent replay attacks.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Replay attacks allow attackers to gain access to information by completing a duplicate transaction which can have been modified before replaying.

As a consequence:

- Authentication requests must include the “nonce” parameter. The value is passed through unmodified from the Authentication Request to the ID Token. If present in the ID Token, Client applications must verify that the “nonce” claim value is equal to the value of the “nonce” parameter sent in the Authentication Request.
- Access token must have short expiration time (Cf. API2-R2).
- In case of sensitive API requests, a one-time-password mechanism or a digital signature-based mechanism should be implemented.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **ID**：API6-R1
 **规则名称**：重放攻击
 **规则摘要**：必须实施安全措施以防止重放攻击。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 重放攻击允许攻击者通过重复提交一笔交易来获取信息，而该交易在被重放前甚至可能已被修改。

因此：

- 认证请求必须包含 “nonce” 参数。该值必须从 Authentication Request 原样传递到 ID Token。如果 ID Token 中存在该字段，客户端应用必须验证其中的 “nonce” claim 是否与 Authentication Request 中发送的 “nonce” 参数值一致。
- Access token 必须有较短的过期时间（参见 API2-R2）。
- 对于敏感 API 请求，应实施一次性密码机制或基于数字签名的机制。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

# Authorization

## 1) API1-R1

**Page Title**: API1-R1
 **ID**: API1-R1
 **Rule Name**: Fined-grained authorizations
 **Rule Summary**: Coarse-grained authorizations must be implemented at the API Management Platform level (API provider). Fined-grained authorizations must be defined and managed at the business service provider’s level.
 **Reference**: [blank]
 **Creation Date**: 20 Mar 2015
 **Last reviewed date**: 01 Dec 2021

**Definition**:
 Authorizations must be verified at every request.

Scopes define coarse-grained authorizations, i.e. the rights to use a service (API). Scopes are managed per Client application at the API Management Platform level (API provider). Fined-grained authorizations define the rights to use business services or data. Fined-grained authorizations are defined and managed at the business service provider’s level. Fined-grained authorizations must be based on the end requesters identities (APIs are requested on behalf of a user and/or a Client application) and the consumers identities of the technical components which transmit the requests.

The recommended solution is based on using access tokens as they can be verified by the API Provider:

- At the API Management Platform level (i.e. API provider):
  - Requesters’ identities (user and/or Client application) must be retrieved thanks to access tokens and/or Client authentication.
  - In case of an access token, the API Provider retrieves the corresponding identity using it to call the UserInfo endpoint.
  - The API Provider verifies Client application’s authorizations to request an API according to scopes.
- Then user’s and Client application’s identities are transmitted by the API Management Platform to business APIs (Business service provider) ensuring their confidentiality and integrity.
- At each business API level, before granting access:
  - The business API must verify whether the request comes from an authorized consumer (here, the API Management Platform).
  - The business API must verify whether the user’s and Client application’s identities transmitted by the API Management Platform are consistent with any similar identifiers present in the data of the request.
  - The business API must verify whether the Client application is authorized to request data. Moreover, if a user’s identity is transmitted or present in the data of the request, the business API must verify whether the Client application is authorized to request data on behalf of the user.
  - The business API must verify whether the user is authorized to request data according to his rights.

**Keywords**: [not visible in screenshot]
 **Review Status**: [not visible in screenshot]
 **Status comments**: [not visible in screenshot]

**中文翻译**
 **页面标题**：API1-R1
 **ID**：API1-R1
 **规则名称**：细粒度授权
 **规则摘要**：粗粒度授权必须在 API 管理平台层面（API provider）实现；细粒度授权必须在业务服务提供方层面定义和管理。
 **参考**：[空白]
 **创建日期**：2015年03月20日
 **最近审查日期**：2021年12月01日

**定义**：
 每一次请求都必须进行授权校验。

Scope 定义的是粗粒度授权，即使用某个服务（API）的权限。Scope 在 API 管理平台层面（API provider）按客户端应用进行管理。细粒度授权定义的是使用业务服务或数据的权限。细粒度授权必须在业务服务提供方层面定义和管理，并且必须基于最终请求者身份（API 是代表用户和/或客户端应用发起的）以及传递请求的技术组件消费者身份。

推荐方案是使用 access token，因为 API Provider 可以校验它：

- 在 API 管理平台层面（即 API provider）：
  - 必须通过 access token 和/或客户端认证获取请求者身份（用户和/或客户端应用）。
  - 如果是 access token，API Provider 通过调用 UserInfo endpoint 来获取对应身份。
  - API Provider 根据 scope 校验客户端应用是否有权限请求该 API。
- 然后，API 管理平台将用户和客户端应用的身份传递给业务 API（Business service provider），并确保这些身份信息的保密性和完整性。
- 在每个业务 API 层面，在授权通过前：
  - 业务 API 必须验证该请求是否来自授权消费者（这里指 API 管理平台）。
  - 业务 API 必须验证 API 管理平台传递过来的用户身份和客户端应用身份，是否与请求数据中存在的类似标识一致。
  - 业务 API 必须验证客户端应用是否有权请求该数据。此外，如果请求中传递了用户身份或数据中存在用户身份，业务 API 还必须验证客户端应用是否有权代表该用户请求数据。
  - 业务 API 必须验证用户本身是否有权根据其权限请求该数据。

**关键词**：[截图中不可见]
 **审查状态**：[截图中不可见]
 **状态说明**：[截图中不可见]

------

## 2) API3-R2

**Page Title**: API3-R2
 **ID**: API3-R2
 **Rule Name**: Secret data
 **Rule Summary**: APIs must not expose secret data.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 APIs must not expose secret data.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: To be changed when we will be able to manage authentication and authorization through data classification.

**中文翻译**
 **页面标题**：API3-R2
 **ID**：API3-R2
 **规则名称**：敏感秘密数据
 **规则摘要**：API 不得暴露 secret data。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 API 不得暴露秘密数据。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：当未来能够通过数据分级来管理认证和授权时，此内容将调整。

------

## 3) API3-R3

**Page Title**: API3-R3
 **ID**: API3-R3
 **Rule Name**: Technical API
 **Rule Summary**: Technical APIs must not be exposed over internet.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Technical APIs (i.e. APIs that gives access on technical infrastructure assets (for example, logs viewing, technical inventories, CI/CD functions, Kubernetes APIs...) or APIs that allows functional or technical administration features (for example, content management system)) must not be exposed over internet.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API3-R3
 **ID**：API3-R3
 **规则名称**：技术类 API
 **规则摘要**：技术类 API 不得暴露到互联网。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 技术类 API（即能够访问技术基础设施资产的 API，例如日志查看、技术资产清单、CI/CD 功能、Kubernetes API 等；或者允许进行功能或技术管理操作的 API，例如内容管理系统）不得暴露到互联网。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 4) API5-R1

**Page Title**: API5-R1
 **ID**: API5-R1
 **Rule Name**: Scope
 **Rule Summary**: Coarse-grained authorizations must be implemented in accordance with data confidentiality and integrity requirements and least privilege principle.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Authorizations must be verified for every request.

Scopes or audience define coarse-grained authorizations, i.e. the rights to use a service (API). They are managed at the API Management Platform level (API provider).

One scope can give access to several APIs. The API Management Platform verifies Client applications authorizations to request APIs according to scopes.

Scopes are defined according to:

- Data confidentiality and integrity requirements: all the APIs in a same scope must have the same confidentiality and integrity requirements. As a consequence, scopes and authentication levels are linked: sensitive APIs will require strong authentication methods. In a nutshell the minimum level of authentication for requesting an API is determined by scopes. Moreover, this minimum level can be elevated depending on rules based on context (for example, IP address of the requester, geolocation, last date of connection...).
- Least privilege principle: To limit the list of scopes per Client application and comply with the least privilege principle, it is recommended that a Client application’s identifier (Client_id) is dedicated to only one version of a Client application software when scopes has deeply changed from one version to another.
- Roles: Role-Based Access Control (RBAC) authorization model could be used to decouple the Clients applications and the assigned roles used in API to check whether that role is authorized to invoke that particular API.

For example, in a banking application, the user wishes to transfer funds from one account to another, which is deemed a sensitive transaction. To perform the action, the application requires a sensitive scope, say “transfer_funds”. If the access token that the user currently has doesn’t include this scope, which the application knows because it knows the set of scopes it requested in the initial authentication process, then the application will perform another authentication request adding the new required scope in its request. Because of the sensitive scope “transfer_funds”, the rules on the OpenID provider will require the user to authenticate using a multi-factor method.

**Keywords**: [not visible in screenshot]
 **Review Status**: [not visible in screenshot]
 **Status comments**: [not visible in screenshot]

**中文翻译**
 **页面标题**：API5-R1
 **ID**：API5-R1
 **规则名称**：Scope
 **规则摘要**：粗粒度授权必须根据数据的保密性、完整性要求以及最小权限原则来实现。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 每一次请求都必须进行授权校验。

Scope 或 audience 定义的是粗粒度授权，即使用某个服务（API）的权限。它们在 API 管理平台层面（API provider）进行管理。

一个 scope 可以赋予访问多个 API 的权限。API 管理平台根据 scope 校验客户端应用是否有权请求某个 API。

Scope 的定义应基于：

- 数据的保密性和完整性要求：同一个 scope 下的所有 API 必须具有相同的保密性和完整性要求。因此，scope 与认证级别是关联的：敏感 API 需要更强的认证方式。简言之，请求某个 API 所需的最低认证级别由 scope 决定。此外，这一最低级别还可以根据上下文规则继续提升（例如请求者的 IP 地址、地理位置、最近一次连接时间等）。
- 最小权限原则：为了减少每个客户端应用持有的 scope 列表并符合最小权限原则，建议当某个客户端应用软件不同版本之间 scope 发生较大变化时，为每个版本使用独立的 Client_id。
- 角色：可以使用基于角色的访问控制（RBAC）授权模型，将客户端应用与 API 中实际使用的角色解耦，从而检查某个角色是否被授权调用特定 API。

例如，在一个银行应用中，用户希望把资金从一个账户转到另一个账户，这被视为敏感交易。要执行这一操作，应用需要一个敏感 scope，比如 “transfer_funds”。如果用户当前持有的 access token 中不包含这个 scope，而应用知道这一点，因为它知道在初始认证过程中曾请求过哪些 scopes，那么应用会再次发起认证请求，并在请求中加入新的必需 scope。由于 “transfer_funds” 是敏感 scope，OpenID provider 上的规则会要求用户采用多因素认证方式。

**关键词**：[截图中不可见]
 **审查状态**：[截图中不可见]
 **状态说明**：[截图中不可见]

------

# Protection

## 1) API2-R6

**Page Title**: API2-R6
 **ID**: API2-R6
 **Rule Name**: OpenID Connect Authorization Code Grant specifics
 **Rule Summary**: OpenID Connect Authorization Code Grant flow must be secure (redirect_uri control, user-agent choice, PKCE).
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Redirect_uri:

- This parameter tells the OpenID Provider where to send the user back to after authentication.
- The domain name of the redirection endpoint Uri must be under the control of the Client application’s owner. The Redirection endpoint Uri must be fully-qualified, unique (no list of URIs) and explicit (no wildcards, no regular expressions...).
- The Redirection endpoint Uri must be controlled by the authorization endpoint for every request (it must match one of the pre-registered Redirect URIs), and by the Client application (it must be identical with the redirect URI sent in the Authentication request).

User-agent for native applications:

- The user-agent can be an external browser or an embedded browser (e.g. webview).
- An external browser can be launched to handle the user’s interaction with the login screen so that the native application itself does not see the user’s credentials. The browser can also display important security information related to the currently loaded page (for example, invalid or expired certificates), whereas embedded browsers often do not. An external browser can use the system cookie store, so it is able to take advantage of already-active sessions (SSO functionality). But external browser vulnerabilities could expose the user’s credentials.

A WebView is single-process, so any security vulnerability in the renderer engine practically grants the malicious code the same rights as your application has: an arbitrary third-party content into a WebView could exploit some security vulnerability of the rendering engine and gains control over it.

As a consequence, the user-agent choice is a trade-off:

- In case of third-party owned Client applications, an external browser must be launched as user-agent.
- On the contrary, because the content inside the login screen is trusted as provided by BNPP itself, in case of BNPP owned Client applications, embedded browsers (e.g. webviews) must be used. If possible, JavaScript should be disabled.
- The OpenID Provider must verify that the user-agent is not a fake one and is consistent with the Client application’s ownership.

Proof Key for Code Exchange (PKCE):

- In case of untrusted Client applications, to mitigate authorization code interception scenarios, the Proof Key for Code Exchange (PKCE) extension must be used.
- When using the PKCE extension, with each authentication request, the Client application creates a cryptographically random code verifier. Two parameters, a code challenge and a code challenge method, are added to the initial Client application authentication request. The challenge method must be “S256” (i.e. the code challenge is a base 64 encoding of the SHA2 256 bitd hash of the code verifier).
- When receiving the request, the OpenID provider notes the code challenge and method and returns the authorization code as usual.
- To convert the received authorization code to a token, the Client application must present its original code verifier along with the authorization code. Using the code challenge and method, the OpenID provider will validate the code challenge before returning valid access and refresh tokens.

**Keywords**: [not visible in screenshot]
 **Review Status**: [not visible in screenshot]
 **Status comments**: [not visible in screenshot]

**中文翻译**
 **页面标题**：API2-R6
 **ID**：API2-R6
 **规则名称**：OpenID Connect Authorization Code Grant 特定要求
 **规则摘要**：OpenID Connect Authorization Code Grant 流程必须安全（redirect_uri 控制、user-agent 选择、PKCE）。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 Redirect_uri：

- 该参数用于告诉 OpenID Provider 在认证完成后将用户重定向到哪里。
- 重定向端点 URI 的域名必须由客户端应用的所有者控制。该 Redirection endpoint URI 必须是完整限定的、唯一的（不能是一组 URI 列表），并且必须显式定义（不能使用通配符、正则表达式等）。
- 对于每一次请求，Redirection endpoint URI 都必须同时受到授权端点和客户端应用的控制：授权端点必须验证其是否匹配某个预先注册的 Redirect URI，而客户端应用也必须验证其与 Authentication Request 中发送的 redirect URI 完全一致。

原生应用的 User-agent：

- User-agent 可以是外部浏览器，也可以是嵌入式浏览器（例如 WebView）。
- 可以启动外部浏览器来承载用户与登录界面的交互，这样原生应用本身就看不到用户凭证。浏览器还可以展示当前页面的重要安全信息（例如证书无效或过期），而嵌入式浏览器通常做不到。外部浏览器还能使用系统 cookie 存储，因此可以利用已有会话（SSO 功能）。但外部浏览器的漏洞也可能暴露用户凭证。

WebView 是单进程的，因此只要渲染引擎存在安全漏洞，恶意代码几乎就会获得与你的应用相同的权限：任意第三方内容只要进入 WebView，就可能利用渲染引擎漏洞并取得控制权。

因此，user-agent 的选择本质上是一种权衡：

- 对于第三方拥有的客户端应用，必须使用外部浏览器作为 user-agent。
- 相反，对于 BNPP 自有客户端应用，由于登录界面的内容被视为由 BNPP 自身提供并可信，因此必须使用嵌入式浏览器（例如 WebView）。如有可能，应禁用 JavaScript。
- OpenID Provider 必须验证该 user-agent 不是伪造的，并且与客户端应用的归属一致。

Proof Key for Code Exchange (PKCE)：

- 对于不可信客户端应用，为缓解 authorization code 被拦截的风险，必须使用 PKCE 扩展。
- 使用 PKCE 时，客户端应用在每次认证请求中都要生成一个密码学随机的 code verifier。然后在初始认证请求中加入两个参数：code challenge 和 code challenge method。Challenge method 必须为 “S256”（即 code challenge 是对 code verifier 做 SHA2 256 位哈希后再进行 base64 编码的结果）。
- OpenID provider 接收到请求后，会记录 code challenge 及其方法，并像往常一样返回 authorization code。
- 客户端应用在将接收到的 authorization code 换成 token 时，必须提交原始的 code verifier。OpenID provider 会利用之前记录的 code challenge 和 method 对其进行校验，校验成功后才返回有效的 access token 和 refresh token。

**关键词**：[截图中不可见]
 **审查状态**：[截图中不可见]
 **状态说明**：[截图中不可见]

------

## 2) API4-R1

**Page Title**: API4-R1
 **ID**: API4-R1
 **Rule Name**: Rate limiting
 **Rule Summary**: Rate limiting must be implemented per API.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 The API Provider must limit the number of requests per time period per Client application (“Client_id”), per Client application and user, or per IP, depending on the context.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API4-R1
 **ID**：API4-R1
 **规则名称**：限流
 **规则摘要**：必须按 API 实施 rate limiting。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 API Provider 必须根据上下文，按客户端应用（“Client_id”）、按客户端应用加用户、或按 IP，在单位时间内限制请求次数。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 3) API7-R1

**Page Title**: API7-R1
 **ID**: API7-R1
 **Rule Name**: Cross-Origin Resource Sharing (CORS)
 **Rule Summary**: In case of APIs expecting to be accessed from browser, a proper CORS policy must be implemented.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 12 May 2022

**Definition**:
 APIs expecting to be accessed from browser must implement a proper CORS policy. CORS is a security mechanism that allows a web page from one domain to access a resource hosted by a different domain (cross-domain request). As such, CORS work in correlation with the Same-Origin Policy (SOP) implemented in modern browsers.

- The API Provider must have full control over whether to allow a request or not depending on the origin of the request: a whitelisted set of authorized origins must be maintained.
- Responses containing “Access-Control-Allow-Origin: *” header must be prohibited.
- Responses containing “Access-Control-Allow-Credentials” tag in header must be prohibited.
- In case of CORS preflight requests, the API Provider must specify which methods are allowed for cross-origin sites to use in the header “Access-Control-Allow-Methods”.

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API7-R1
 **ID**：API7-R1
 **规则名称**：跨域资源共享（CORS）
 **规则摘要**：对于预期会被浏览器访问的 API，必须实施正确的 CORS 策略。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2022年05月12日

**定义**：
 凡是预期会被浏览器访问的 API，都必须实现正确的 CORS 策略。CORS 是一种安全机制，它允许来自一个域的网页访问托管在另一个域上的资源（跨域请求）。因此，CORS 与现代浏览器中实现的同源策略（Same-Origin Policy, SOP）是协同工作的。

- API Provider 必须对是否允许某个请求拥有完全控制权，并且必须根据请求来源进行判断：必须维护一份被允许的 origin 白名单。
- 严禁返回包含 “Access-Control-Allow-Origin: *” header 的响应。
- 严禁返回包含 “Access-Control-Allow-Credentials” header 的响应。
- 对于 CORS 预检请求，API Provider 必须通过 “Access-Control-Allow-Methods” header 明确声明允许跨域站点使用哪些方法。

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：[空白]

------

## 4) API7-R2

**Page Title**: API7-R2
 **ID**: API7-R2
 **Rule Name**: Errors management
 **Rule Summary**: Errors must be securely managed (no technical details are to be specified).
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:

- Standard HTTP status codes must be used.
- To prevent technical details (e.g. exception traces) and other valuable information from being sent back to attackers, the API Provider must enforce all API response payload schemas including error responses.
- In case of malformed request, no response must be sent back to the requester.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API7-R2
 **ID**：API7-R2
 **规则名称**：错误管理
 **规则摘要**：错误必须被安全处理（不得暴露技术细节）。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：

- 必须使用标准 HTTP 状态码。
- 为防止将技术细节（例如异常堆栈）和其他有价值信息返回给攻击者，API Provider 必须对所有 API 响应负载模式进行强制约束，包括错误响应。
- 对于格式错误的请求，不得向请求方返回响应。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 5) API7-R4

**Page Title**: API7-R4
 **ID**: API7-R4
 **Rule Name**: Predictable Resource Location attack
 **Rule Summary**: Security measures must be implemented to prevent Predictable Resource Location attack.
 **Reference**: [blank]
 **Creation Date**: 16 May 2024
 **Last reviewed date**: 16 May 2024

**Definition**:
 Predictable Resource Location is a brute force attack method used to uncover hidden website content or functionalities like APIs. Attackers can also reverse engineer APIs by examining Client application’s code (in case of mobile or browser-based applications) or simply monitor and analyze communications to be able to call APIs by robots. Despite Web Application Firewall (WAF), it is crucial that the API is developed with safeguards in place:

- API must be only visible for the Client applications they intend to: for example, extranet-facing API must not be accessible from Internet.
- HTTP Browsing must be disabled on directories that are not explicitly listed to be browsable.
- Internet API endpoints (the API Provider itself but also the OpenID Provider) must not be indexed by crawlers.
- Classic API paths like /api, /api/v1, /apis.json must be avoided because attackers could easily guess them.

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**: rendre non predictible les ids des objets ?

**中文翻译**
 **页面标题**：API7-R4
 **ID**：API7-R4
 **规则名称**：可预测资源位置攻击
 **规则摘要**：必须实施安全措施防止 Predictable Resource Location attack。
 **参考**：[空白]
 **创建日期**：2024年05月16日
 **最近审查日期**：2024年05月16日

**定义**：
 Predictable Resource Location 是一种暴力猜测攻击方法，用来发现网站中隐藏的内容或功能，例如 API。攻击者也可以通过检查客户端应用代码（例如移动应用或浏览器应用），或者通过监控与分析通信流量，来逆向出 API，从而让机器人调用这些 API。即使有 Web Application Firewall（WAF），API 在开发时仍必须具备以下防护措施：

- API 只能对其目标客户端应用可见：例如，面向 extranet 的 API 不得从互联网直接访问。
- 对未被明确允许浏览的目录，必须禁用 HTTP 浏览功能。
- 面向互联网的 API 端点（包括 API Provider 本身以及 OpenID Provider）不得被爬虫索引。
- 应避免使用经典 API 路径，例如 /api、/api/v1、/apis.json，因为攻击者很容易猜到这些路径。

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：是否还应让对象 ID 也变得不可预测？

------

## 6) API7-R5

**Page Title**: API7-R5
 **ID**: API7-R5
 **Rule Name**: Cross-Site Request Forgery (CSRF) attack
 **Rule Summary**: Security measures must be implemented to prevent Cross-Site Request Forgery (CSRF) attacks.
 **Reference**: [blank]
 **Creation Date**: 16 May 2024
 **Last reviewed date**: 16 May 2024

**Definition**:

- The “state” parameter must be used in Authentication requests. This parameter must be one-time use, session-specific and unpredictable.

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**: Il manque la prise en compte du CSRF pour API gerer par cookie

**中文翻译**
 **页面标题**：API7-R5
 **ID**：API7-R5
 **规则名称**：跨站请求伪造（CSRF）攻击
 **规则摘要**：必须实施安全措施防止 CSRF 攻击。
 **参考**：[空白]
 **创建日期**：2024年05月16日
 **最近审查日期**：2024年05月16日

**定义**：

- 在认证请求中必须使用 “state” 参数。该参数必须是一次性的、与会话绑定的、且不可预测。

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：当前文档还缺少对基于 cookie 管理 API 时 CSRF 风险的考虑。

------

## 7) API7-R6

**Page Title**: API7-R6
 **ID**: API7-R6
 **Rule Name**: Vulnerability testing for internet-facing API
 **Rule Summary**: Vulnerability tests must be performed.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Vulnerability tests must be performed before the release of major versions of APIs, and every month in production. Critical level vulnerabilities must be fixed before going live.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API7-R6
 **ID**：API7-R6
 **规则名称**：面向互联网 API 的漏洞测试
 **规则摘要**：必须执行漏洞测试。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 在 API 重大版本发布前，必须执行漏洞测试；在生产环境中，必须每月执行一次。所有 Critical 级别漏洞都必须在上线前修复。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 8) API7-R7

**Page Title**: API7-R7
 **ID**: API7-R7
 **Rule Name**: Penetration testing for internet-facing API
 **Rule Summary**: Penetration tests must be performed.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:

- Penetration tests must be performed before each release of major versions of APIs. Critical level vulnerabilities must be fixed before going live.
- In case of confidentiality or integrity level equal to 3 or higher, penetration tests must be performed at least once a year, else penetration tests must be performed at least every two years.
- Penetration tests must include at least a gray-box type of test.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API7-R7
 **ID**：API7-R7
 **规则名称**：面向互联网 API 的渗透测试
 **规则摘要**：必须执行渗透测试。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：

- 每次 API 重大版本发布前都必须执行渗透测试。所有 Critical 级别漏洞都必须在上线前修复。
- 如果保密性或完整性级别大于等于 3，则至少每年执行一次渗透测试；否则至少每两年执行一次。
- 渗透测试至少必须包含 gray-box 类型测试。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

## 9) API8-R1

**Page Title**: API8-R1
 **ID**: API8-R1
 **Rule Name**: Parameters attack
 **Rule Summary**: Security measures must be implemented to prevent Parameters attacks (parameters brute force attack, command injection, SQL injection...).
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 Parameters attacks (parameters brute force attack, command injection, SQL injection...) involve submitting unexpected data to exploit weaknesses in API and see if it breaks leaking data or causing a system failure. Despite Web Application Firewall (WAF) and Intrusion Detection/Prevention Systems (IDS/IPS), it is crucial that the API is developed with safeguards in place to prevent malicious use when Client applications make API calls. It is on the API developer to ensure that the API properly validates all input from the user made during any calls to prevent this from occurring.

- The API Management Platform must perform syntactical input validation of each request (for example, based on detailed swagger definitions for the API) to avoid rogue requests to penetrate the information system and to be processed by the business service provider:
  - HTTP verbs (GET, POST, PUT...),
  - Content types,
  - Paths (i.e. API endpoints),
  - Path parameters, query parameters,
  - Requests and responses body structure,
  - Data types (integer, float, string, date, boolean...),
  - Data length (minimum and maximum),
  - Data format: minimum and maximum value for a number, pattern for an email, list of authorized values, date format...,
  - Free-form text generic sanitization (authorized characters whitelisting).
- The Business services must perform syntactical and semantic input validation of each request to only allow valid values for each input parameter:
  - The same controls as the API Management Platform must be performed again.
  - Business specific controls (for example, contract identifier format, card number format...) must be performed.
  - Free-form text specific sanitization (scrubbing user input of HTML tags (if they are not necessary for the application well-functioning), JavaScript tags, SQL statements...). Although, such a sanitization must have been performed prior on the Client application and on Web Application Firewalls, it must be performed again because the previous treatments should have been by-passed by an attacker

**Keywords**: [not visible in screenshot]
 **Review Status**: [not visible in screenshot]
 **Status comments**: [not visible in screenshot]

**中文翻译**
 **页面标题**：API8-R1
 **ID**：API8-R1
 **规则名称**：参数攻击
 **规则摘要**：必须实施安全措施以防止参数攻击（参数暴力破解、命令注入、SQL 注入等）。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 参数攻击（例如参数暴力破解、命令注入、SQL 注入等）是指提交异常或恶意数据，利用 API 的薄弱点，看看是否能导致数据泄露或系统故障。即使已经有 Web Application Firewall（WAF）和 Intrusion Detection/Prevention Systems（IDS/IPS），API 在开发时仍必须具备防护措施，以防止客户端应用调用 API 时被恶意利用。API 开发者必须确保 API 对所有用户输入都进行适当校验，以避免上述问题发生。

- API 管理平台必须对每个请求执行语法级输入校验（例如基于该 API 的详细 Swagger 定义），以避免异常请求穿透信息系统并被业务服务提供方处理：
  - HTTP 方法（GET、POST、PUT 等）
  - Content-Type
  - 路径（即 API 端点）
  - Path 参数、Query 参数
  - 请求体和响应体结构
  - 数据类型（整数、浮点、字符串、日期、布尔等）
  - 数据长度（最小值和最大值）
  - 数据格式：例如数字的最小值和最大值、邮箱格式模式、允许值列表、日期格式等
  - 对自由文本做通用清洗（允许字符白名单）
- 业务服务必须再次对每个请求执行语法级和语义级输入校验，只允许每个输入参数出现合法值：
  - API 管理平台做过的同类校验必须再次执行。
  - 必须执行业务特定校验（例如合同编号格式、卡号格式等）。
  - 必须执行自由文本的特定清洗（如去除用户输入中的 HTML 标签〔如果应用正常运行不需要这些标签〕、JavaScript 标签、SQL 语句等）。即使这些清洗此前应已在客户端应用和 Web Application Firewall 上完成，也必须再次执行，因为攻击者可能已经绕过前面的处理。

**关键词**：[截图中不可见]
 **审查状态**：[截图中不可见]
 **状态说明**：[截图中不可见]

------

## 10) API8-R2

**Page Title**: API8-R2
 **ID**: API8-R2
 **Rule Name**: Secure coding
 **Rule Summary**: A code audit must be performed.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 A code audit (Static Application Security Testing (SAST)) must be performed before going live.

Critical and high level vulnerabilities must be fixed before going live.

**Keywords**: [blank]
 **Review Status**: IN PROGRESS
 **Status comments**: Critical and high level vulnerabilities must be fixed before going live.

**中文翻译**
 **页面标题**：API8-R2
 **ID**：API8-R2
 **规则名称**：安全编码
 **规则摘要**：必须执行代码审计。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 上线前必须执行代码审计（静态应用安全测试，Static Application Security Testing，SAST）。

所有 Critical 和 High 级别漏洞都必须在上线前修复。

**关键词**：[空白]
 **审查状态**：进行中
 **状态说明**：所有 Critical 和 High 级别漏洞都必须在上线前修复。

------

## 11) API10-R1

**Page Title**: API10-R1
 **ID**: API10-R1
 **Rule Name**: Traceability
 **Rule Summary**: A trace of all the interactions with the system shall be retained. Traces must be protected against any malicious modification.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 At each component level (API Provider, Business service provider, OpenID Provider...):

- All login attempts (legitimate or not), access control failures, personal data changes, sensitive activities, consents, server-side input validation failures must be logged with sufficient context (action performed, origin (natural person, technical equipment, computer program, source IP address, etc.), moment of realization) to identify suspicious or malicious activities.
- Traces must be protected against any malicious modification. Moreover, traces must not stay locally more than a day. Traces must consolidated in real-time in a central repository to alert in case of abnormal behavior. The traces must be kept at least 12 months.
- No secret data (for example, passwords, card numbers) must be recorded in the traces. As an exception, masked card numbers according to local banking recommendations (or at least PCI-DSS recommendations) are allowed. No confidential data (for example, personal data) should be recorded in the traces without a definite objective concerning their use a posteriori.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：API10-R1
 **ID**：API10-R1
 **规则名称**：可追溯性
 **规则摘要**：必须保留与系统所有交互的轨迹记录，并且这些记录必须防止被恶意篡改。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 在每一个组件层级（API Provider、Business service provider、OpenID Provider 等）：

- 所有登录尝试（无论合法与否）、访问控制失败、个人数据变更、敏感操作、consent、服务端输入校验失败，都必须记录日志，并包含足够上下文（执行了什么动作、来源〔自然人、技术设备、计算机程序、源 IP 地址等〕、发生时间），以便识别可疑或恶意活动。
- 这些日志轨迹必须防止被恶意篡改。此外，日志不得在本地保存超过一天。日志必须实时汇聚到中央存储库，以便在出现异常行为时发出告警。日志至少必须保留 12 个月。
- 日志中不得记录秘密数据（例如密码、卡号）。作为例外，可以记录按照本地银行业建议（或至少符合 PCI-DSS 建议）进行掩码处理后的卡号。对于机密数据（例如个人数据），如果没有明确的事后使用目的，则不应记录在日志中。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]

------

# Others

## 1) API9-R1

**Page Title**: API9-R1
 **ID**: API9-R1
 **Rule Name**: Governance
 **Rule Summary**: Roles and responsabilities regarding the API must be specified in a registry.
 **Reference**: [blank]
 **Creation Date**: 29 Jun 2021
 **Last reviewed date**: 29 Jun 2021

**Definition**:
 The API Management Platform must maintain a registry of the APIs:

- API IT owner, API Business owner
- API description (functional description, targeted users, handled data, reliability (SLA, RTO, RPO)...),
- API version,
- API status (active, deprecated, inactive),
- Business services and data managed by the API (with a detailed swagger description),
- API data classification (confidentiality, integrity, availability, traceability levels),
- API exposition (internet, extranet, intranet),
- API scope(s),
- Authorized Client applications according to the scope(s).

**Keywords**: [not visible in screenshot]
 **Review Status**: [not visible in screenshot]
 **Status comments**: [not visible in screenshot]

**中文翻译**
 **页面标题**：API9-R1
 **ID**：API9-R1
 **规则名称**：治理
 **规则摘要**：与 API 相关的角色和职责必须在注册表中明确说明。
 **参考**：[空白]
 **创建日期**：2021年06月29日
 **最近审查日期**：2021年06月29日

**定义**：
 API 管理平台必须维护一份 API 注册表，至少包括：

- API IT owner、API Business owner
- API 描述（功能描述、目标用户、处理的数据、可靠性要求〔SLA、RTO、RPO〕等）
- API 版本
- API 状态（active、deprecated、inactive）
- API 管理的业务服务和数据（附详细 Swagger 描述）
- API 数据分级（保密性、完整性、可用性、可追溯性级别）
- API 暴露范围（internet、extranet、intranet）
- API scope(s)
- 根据 scope(s) 被授权的客户端应用

**关键词**：[截图中不可见]
 **审查状态**：[截图中不可见]
 **状态说明**：[截图中不可见]

------

## 2) API9-R2

**Page Title**: [not visible in screenshot]
 **ID**: API9-R2
 **Rule Name**: Unused API
 **Rule Summary**: Any API that is not requested during 6 months should be deactivated.
 **Reference**: [blank]
 **Creation Date**: 12 May 2022
 **Last reviewed date**: 13 May 2022

**Definition**:
 Any API that is not requested for 6 months should be deactivated (or deleted) by the API owner if there is no justification.

**Keywords**: [blank]
 **Review Status**: VALIDATED
 **Status comments**: [blank]

**中文翻译**
 **页面标题**：[截图中不可见]
 **ID**：API9-R2
 **规则名称**：未使用的 API
 **规则摘要**：任何 6 个月内未被调用的 API 都应被停用。
 **参考**：[空白]
 **创建日期**：2022年05月12日
 **最近审查日期**：2022年05月13日

**定义**：
 任何在 6 个月内未被请求的 API，如果没有合理说明，API owner 应将其停用（或删除）。

**关键词**：[空白]
 **审查状态**：已验证
 **状态说明**：[空白]
